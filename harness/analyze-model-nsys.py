#!/usr/bin/env python3
"""CPU-only, read-only Nsight SQLite analysis; exports no raw labels or strings.

Example: python analyze-model-nsys.py --input PRIVATE.sqlite --output summary.json
Phase attribution follows CUDA launch correlation -> same-thread NVTX, never
GPU/CPU timestamp coincidence. Graph membership alone is not a decode label.

Schema 2 keeps every schema 1 section and adds: per-forward instances (one
Qwen36/prefill or Qwen36/decode push/pop range each), request segmentation,
a GPU critical chain per forward (latest-finishing-predecessor walk, so
overlapped kernel durations are never summed), concurrency-share wall
attribution, nested Sampler handling (duplicate "Sampler" or
"Sampler"+"SamplerKernelDispatch" without editing the runtime), layer
attribution (eager launches through same-thread Layer_N NVTX; CUDA graph nodes
through the capture-time NVTX recorded by --cuda-graph-trace=node:nvtx-precapture)
and a ranked list of Nsight Compute candidates by critical-chain time.
Schema 3 adds the per-request (per-scenario) critical path, so warm short
decode, fresh 16K and fresh 120K are never mixed when choosing a target, and
chooses each private Compute instance by phase + shape class + full launch
config inside the request that carries the most critical time for it.
Kernel symbols leave the script only through --private-targets (0600 file in a
0700 directory), never through the public JSON.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile

SCHEMA_VERSION = 3
PHASE_LABELS = {
    "prefill": {"Qwen36/prefill", "Qwen36.forward.prefill", "prefill"},
    "decode": {"Qwen36/decode", "Qwen36.forward.decode", "decode"},
}
TABLES = {
    "kernel": "CUPTI_ACTIVITY_KIND_KERNEL",
    "memcpy": "CUPTI_ACTIVITY_KIND_MEMCPY",
    "memset": "CUPTI_ACTIVITY_KIND_MEMSET",
    "runtime": "CUPTI_ACTIVITY_KIND_RUNTIME",
    "driver": "CUPTI_ACTIVITY_KIND_DRIVER",
    "nvtx": "NVTX_EVENTS",
}
GRAPH_NODE_TABLE = "CUDA_GRAPH_NODE_EVENTS"
# Only these columns are read; absent columns become None. Large final traces
# (hundreds of thousands of kernels) must not be materialised with SELECT *.
COLUMNS = {
    "kernel": ("start", "end", "deviceId", "contextId", "streamId", "globalPid", "correlationId",
               "demangledName", "shortName", "graphId", "graphNodeId", "gridX", "gridY", "gridZ",
               "blockX", "blockY", "blockZ", "staticSharedMemory", "dynamicSharedMemory"),
    "memcpy": ("start", "end", "deviceId", "contextId", "streamId", "globalPid", "correlationId",
               "bytes", "copyKind", "graphNodeId", "graphId"),
    "memset": ("start", "end", "deviceId", "contextId", "streamId", "globalPid", "correlationId",
               "bytes", "graphNodeId", "graphId"),
    "runtime": ("start", "end", "globalTid", "correlationId", "nameId"),
    "driver": ("start", "end", "globalTid", "correlationId", "nameId"),
    "nvtx": ("start", "end", "eventType", "globalTid", "endGlobalTid", "text", "textId"),
}
GRAPH_NODE_COLUMNS = ("start", "end", "globalTid", "graphNodeId", "originalGraphNodeId")
LAYER_RE = re.compile(r"^Layer_(\d{1,4})$")
# Fixed structural NVTX labels from the FreeToken source. Nothing else is
# interpreted and no label text is exported.
REGION_LABELS = {"MHA": "mha", "Sampler": "sampler", "SamplerKernelDispatch": "sampler_dispatch",
                 "Embedding": "embedding", "LMHead": "lm_head"}
REQUEST_LABEL_RE = re.compile(r"^[a-z0-9_]{1,32}$")
UNMEASURED_PREFIXES = ("warmup", "filler")
TOP_CANDIDATES = 16
TOP_REQUEST_CANDIDATES = 8


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(left, right):
    left, right = merged(left), merged(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def spans(rows):
    return [(row["start"], row["end"]) for row in rows]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def stats_ms(values_ns):
    """Distribution of nanosecond values in milliseconds (no sums of overlaps)."""
    if not values_ns:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {"count": len(values_ns), "mean_ms": sum(values_ns) / len(values_ns) / 1e6,
            "p50_ms": percentile(values_ns, .5) / 1e6, "p95_ms": percentile(values_ns, .95) / 1e6,
            "min_ms": min(values_ns) / 1e6, "max_ms": max(values_ns) / 1e6}


def timeline(rows):
    intervals = merged(spans(rows))
    if not intervals:
        return {"event_count": 0, "envelope_ms": 0.0, "activity_union_ms": 0.0,
                "unobserved_inside_envelope_ms": 0.0, "gap_count": 0,
                "gap_p50_ms": None, "gap_p95_ms": None, "gap_max_ms": None}
    gaps = [b[0] - a[1] for a, b in zip(intervals, intervals[1:])]
    envelope = intervals[-1][1] - intervals[0][0]
    active = sum(end - start for start, end in intervals)
    return {"event_count": len(rows), "envelope_ms": envelope / 1e6,
            "activity_union_ms": active / 1e6,
            "unobserved_inside_envelope_ms": (envelope - active) / 1e6,
            "gap_count": len(gaps),
            "gap_p50_ms": percentile(gaps, .5) / 1e6 if gaps else None,
            "gap_p95_ms": percentile(gaps, .95) / 1e6 if gaps else None,
            "gap_max_ms": max(gaps) / 1e6 if gaps else None}


def simultaneous_streams(groups):
    events = []
    for rows in groups:
        for start, end in merged(spans(rows)):
            events.extend(((start, 1), (end, -1)))
    active = peak = total = 0
    previous = None
    # Aggregate equal timestamps, so touching intervals do not create overlap.
    deltas = collections.defaultdict(int)
    for instant, delta in events:
        deltas[instant] += delta
    for instant, delta in sorted(deltas.items()):
        if previous is not None and active >= 2:
            total += instant - previous
        active += delta
        peak = max(peak, active)
        previous = instant
    return total, peak


def category(name):
    # Only fixed category labels leave this script. Arbitrary symbol/NVTX text
    # and StringIds can contain user data, paths, prompt content or API secrets.
    if "attention_gate_fp8" in name or "gate_quant_kernel" in name:
        return "attention_gate_fp8_fused"
    if "prefill_nvfp4" in name or "nvfp4_prefill" in name:
        return "moe_prefill_nvfp4"
    if "fast_index_copy_multi" in name or "mapped_gather" in name:
        return "expert_mapped_host_gather"
    if any(token in name for token in ("_decode_nvfp4_marlin_kernel", "_nvfp4_gemv_splitk_kernel")):
        return "moe_decode_nvfp4"
    if "_nvfp4_gemv_kernel" in name:
        return "nvfp4_linear_gemv"
    if "MainloopSm120" in name and "e4m3" in name:
        return "dense_sm120_fp8_gemm"
    if any(token in name for token in ("gated_delta_rule", "gating_delta_rule", "chunk_fwd_kernel_o",
               "recompute_w_u", "causal_conv1d", "chunk_local_cumsum", "l2norm_fwd")):
        return "gdn_recurrence_conv_chunk"
    if "_layer_norm_fwd_1pass_kernel" in name:
        return "gdn_output_gated_norm"
    if any(token in name for token in ("BatchPrefillWithPagedKVCache", "BatchDecodeWithPagedKVCache",
                                       "PersistentVariableLengthMergeStates")):
        return "full_attention"
    if any(token in name for token in ("BatchQKApplyRotary", "store_kv_cache", "qknorm_rope_store")):
        return "qk_rope_kv_store"
    if any(token in name for token in ("flashinfernormkernels", "rmsnormRMSNorm")):
        return "rmsnorm"
    if any(token in name for token in ("sampling::", "OnlineSoftmax", "TopPSampling", "RadixTopK")):
        return "sampling"
    if "_static_quant_kernel" in name:
        return "static_fp8_activation_quantize"
    if "direct_copy_kernel_cuda" in name:
        return "tensor_layout_dtype_copy"
    if "_lru_ensure_kernel" in name:
        return "expert_lru_metadata"
    if any(token in name for token in ("_router_triton", "moe_sum_reduce", "moe_align_block", "count_and_sort_expert")):
        return "moe_router_alignment_reduction"
    if "act_and_mul" in name:
        return "swiglu"
    if any(token in name for token in ("xmma_gemm", "nvjet_sm120", "cutlass::Kernel2",
                "internal::gemvx", "cublasLt::splitKreduce", "_nvfp4_gemm", "wmma")):
        return "other_dense_gemm_gemv_reduction"
    if any(token in name for token in ("elementwise_kernel", "index_kernel", "FillFunctor", "vectorized_gather")):
        return "other_elementwise_index"
    return "other"


def kernel_summary(rows):
    total = sum(row["end"] - row["start"] for row in rows)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    categories = []
    for label, members in groups.items():
        ns = sum(row["end"] - row["start"] for row in members)
        categories.append({"category": label, "count": len(members), "sum_ms": ns / 1e6,
                           "share_sum_kernel_percent": 100 * ns / total if total else None,
                           "union_ms": duration(spans(members)) / 1e6})
    return {"kernel_count": len(rows), "sum_kernel_ms": total / 1e6,
            "union_kernel_ms": duration(spans(rows)) / 1e6,
            "categories": sorted(categories, key=lambda row: row["sum_ms"], reverse=True)}


class PhaseIndex:
    def __init__(self, rows, labels):
        self.groups = collections.defaultdict(list)
        self.raw = collections.defaultdict(list)
        self.cross_thread_ranges_ignored = 0
        for row in rows:
            phase = next((phase for phase, names in labels.items() if row["name"] in names), None)
            if phase is None:
                continue
            if row.get("endGlobalTid") not in (None, row.get("globalTid")):
                self.cross_thread_ranges_ignored += 1
                continue
            if row.get("globalTid") is None:
                continue
            self.raw[phase].append(row)
            self.groups[(row["globalTid"], phase)].append((row["start"], row["end"]))
        self.index = {}
        for key, intervals in self.groups.items():
            intervals = merged(intervals)
            self.index[key] = ([start for start, _ in intervals], intervals)

    def find(self, row):
        result = set()
        for phase in ("prefill", "decode"):
            starts, intervals = self.index.get((row.get("globalTid"), phase), ([], []))
            index = bisect.bisect_right(starts, row["start"]) - 1
            if index >= 0 and row["end"] <= intervals[index][1]:
                result.add(phase)
        return next(iter(result)) if len(result) == 1 else "unknown"

    def summary(self):
        return {phase: {"range_count": len(self.raw[phase]),
                        "inclusive_cpu_sum_ms": sum(r["end"] - r["start"] for r in self.raw[phase]) / 1e6,
                        "cpu_union_ms": duration(spans(self.raw[phase])) / 1e6}
                for phase in ("prefill", "decode")}


class NestedLookup:
    """Innermost interval containing an instant, for properly nested ranges.

    NVTX push/pop ranges of one thread form a stack, so for a filtered subset
    the innermost containing interval is the last-starting one or one of its
    ancestors. Parent links make the walk exact without scanning.
    """

    def __init__(self, items):
        self.items = sorted(items, key=lambda item: (item[0], -item[1]))
        self.starts = [item[0] for item in self.items]
        self.parent = []
        stack = []
        for index, (start, end, _) in enumerate(self.items):
            while stack and self.items[stack[-1]][1] < end:
                stack.pop()
            self.parent.append(stack[-1] if stack else -1)
            stack.append(index)

    def find(self, instant, end=None):
        index = bisect.bisect_right(self.starts, instant) - 1
        while index >= 0:
            start, stop, value = self.items[index]
            if start <= instant and instant <= stop and (end is None or end <= stop):
                return value
            index = self.parent[index]
        return None


class InstanceIndex:
    """One forward instance per outermost same-thread phase range."""

    def __init__(self, rows, labels):
        per_thread = collections.defaultdict(list)
        self.nested_phase_ranges_ignored = 0
        for row in rows:
            phase = next((phase for phase, names in labels.items() if row["name"] in names), None)
            if phase is None or row.get("globalTid") is None:
                continue
            if row.get("endGlobalTid") not in (None, row.get("globalTid")):
                continue
            per_thread[row["globalTid"]].append((row["start"], row["end"], phase))
        instances = []
        for tid, ranges in per_thread.items():
            ranges.sort(key=lambda item: (item[0], -item[1]))
            last_end = None
            for start, end, phase in ranges:
                if last_end is not None and start >= outer_start and end <= last_end:
                    self.nested_phase_ranges_ignored += 1
                    continue
                outer_start, last_end = start, end
                instances.append({"tid": tid, "start": start, "end": end, "phase": phase})
        instances.sort(key=lambda item: (item["start"], item["tid"]))
        for ordinal, item in enumerate(instances):
            item["id"] = ordinal
        self.instances = instances
        grouped = collections.defaultdict(list)
        for item in instances:
            grouped[item["tid"]].append((item["start"], item["end"], item["id"]))
        self.lookup = {tid: NestedLookup(items) for tid, items in grouped.items()}

    def find(self, row):
        lookup = self.lookup.get(row.get("globalTid"))
        return None if lookup is None else lookup.find(row["start"], row["end"])


class RegionIndex:
    """Same-thread structural NVTX regions: Layer_N, MHA, Sampler, Embedding, LMHead."""

    def __init__(self, rows):
        layers = collections.defaultdict(list)
        regions = collections.defaultdict(lambda: collections.defaultdict(list))
        self.label_counts = collections.Counter()
        self.layer_has_mha = collections.defaultdict(bool)
        self.sampler_rows = []
        for row in rows:
            tid = row.get("globalTid")
            if tid is None or row.get("endGlobalTid") not in (None, tid):
                continue
            name = row["name"]
            match = LAYER_RE.match(name)
            if match:
                layers[tid].append((row["start"], row["end"], int(match.group(1))))
                self.label_counts["layer"] += 1
            elif name in REGION_LABELS:
                kind = REGION_LABELS[name]
                regions[tid][kind].append((row["start"], row["end"], kind))
                self.label_counts[kind] += 1
                if kind in ("sampler", "sampler_dispatch"):
                    self.sampler_rows.append(dict(row, kind=kind))
        self.layers = {tid: NestedLookup(items) for tid, items in layers.items()}
        self.regions = {tid: {kind: NestedLookup(items) for kind, items in kinds.items()}
                        for tid, kinds in regions.items()}
        # Layer kind: a Layer_N that encloses an MHA range is a full-attention
        # layer; the remaining hybrid layers carry no MHA range.
        for tid, kinds in regions.items():
            lookup = self.layers.get(tid)
            if lookup is None:
                continue
            for start, end, _ in kinds.get("mha", []):
                layer = lookup.find(start, end)
                if layer is not None:
                    self.layer_has_mha[layer] = True
        self.observed_layers = sorted({value for items in layers.values() for _, _, value in items})

    def find(self, tid, instant):
        layer = self.layers[tid].find(instant) if tid in self.layers else None
        kinds = self.regions.get(tid, {})
        region = None
        for kind in ("sampler_dispatch", "sampler", "lm_head", "embedding", "mha"):
            if kind in kinds and kinds[kind].find(instant) is not None:
                region = "sampler" if kind == "sampler_dispatch" else kind
                break
        return layer, region

    def layer_kind(self, layer):
        if layer is None:
            return "outside_layers"
        return "mha" if self.layer_has_mha.get(layer) else "no_mha"

    def sampler_summary(self, instances):
        rows = self.sampler_rows
        if not rows:
            return {"mode": "absent", "range_count": 0}
        by_tid = collections.defaultdict(list)
        for row in rows:
            by_tid[row["globalTid"]].append(row)
        nested_same = nested_dispatch = 0
        outer = []
        for tid, items in by_tid.items():
            items.sort(key=lambda row: (row["start"], -row["end"]))
            current = None
            for row in items:
                if current is not None and row["start"] >= current["start"] and row["end"] <= current["end"]:
                    if row["kind"] == "sampler_dispatch":
                        nested_dispatch += 1
                    else:
                        nested_same += 1
                    continue
                current = row
                outer.append(row)
        if nested_same:
            mode = "nested_same_label"
        elif nested_dispatch:
            mode = "outer_plus_dispatch"
        else:
            mode = "single"
        per_instance = collections.Counter()
        lookups = collections.defaultdict(list)
        for item in instances:
            lookups[item["tid"]].append((item["start"], item["end"], item["id"]))
        lookups = {tid: NestedLookup(items) for tid, items in lookups.items()}
        for row in rows:
            lookup = lookups.get(row["globalTid"])
            instance = lookup.find(row["start"], row["end"]) if lookup else None
            per_instance[instance] += 1
        outer_ns = sorted((row["start"], row["end"] - row["start"]) for row in outer)
        durations = [value for _, value in outer_ns]
        return {
            "mode": mode, "range_count": len(rows), "outer_call_count": len(outer),
            "nested_same_label_ranges_ignored": nested_same,
            "nested_dispatch_ranges_ignored": nested_dispatch,
            "inclusive_cpu_sum_all_ranges_ms": sum(r["end"] - r["start"] for r in rows) / 1e6,
            "cpu_union_ms": duration(spans(rows)) / 1e6,
            "outer_calls_cpu_ms": stats_ms(durations),
            "first_outer_call_ms": durations[0] / 1e6 if durations else None,
            "after_first_outer_calls_cpu_ms": stats_ms(durations[1:]),
            "ranges_per_forward_instance_distribution": {str(count): n for count, n in sorted(
                collections.Counter(v for k, v in per_instance.items() if k is not None).items())},
            "ranges_outside_forward_instances": per_instance.get(None, 0),
        }


def api_class(name):
    for token, label in (("GraphLaunch", "graph_launch"), ("LaunchKernel", "kernel_launch"),
             ("LaunchCooperativeKernel", "kernel_launch"), ("Memcpy", "memcpy"),
             ("Memset", "memset"), ("StreamSynchronize", "stream_synchronize"),
             ("EventSynchronize", "event_synchronize"), ("DeviceSynchronize", "device_synchronize"),
             ("StreamWaitEvent", "stream_wait_event")):
        if token in name:
            return label
    return "other"


class CorrelationIndex:
    def __init__(self, apis, phases):
        self.index = collections.defaultdict(list)
        for row in apis:
            if row.get("correlationId") is None or row.get("globalTid") is None:
                continue
            row["phase"] = phases.find(row)
            # Nsight's packed globalTid stores the OS thread in its low 24 bits;
            # CUPTI globalPid has those bits cleared. Keep process scope: CUDA
            # correlation IDs from different processes can be identical.
            pid = row["globalTid"] & ~((1 << 24) - 1)
            self.index[(pid, row["correlationId"])].append(row)

    def candidates(self, gpu):
        rows = self.index.get((gpu.get("globalPid"), gpu.get("correlationId")), [])
        return [row for row in rows if row["start"] <= gpu["start"]]

    def find(self, gpu):
        candidates = self.candidates(gpu)
        if not candidates:
            return "unknown", "missing_launch_correlation"
        # Runtime and driver may both describe one launch. A reused correlation
        # with conflicting phases is left unassigned instead of guessed.
        known = {row["phase"] for row in candidates if row["phase"] != "unknown"}
        if len(known) > 1:
            return "unknown", "ambiguous_phase"
        if len(known) == 1:
            return next(iter(known)), "nvtx_launch_correlation"
        return "unknown", "launch_outside_known_phase"


def copy_direction(row, enums):
    name = enums.get(row.get("copyKind"), "")
    if name.endswith(("_HTOD", "_HTOA")):
        return "h2d"
    if name.endswith(("_DTOH", "_ATOH")):
        return "d2h"
    if name.endswith(("_DTOD", "_ATOD", "_DTOA", "_ATOA")):
        return "d2d"
    if name.endswith("_HTOH"):
        return "h2h"
    if name.endswith("_PTOP"):
        return "peer"
    return "unknown"


def read_rows(conn, table, tables, columns=None):
    if table not in tables:
        return []
    # table is from the static allowlist or fixed schema names, not user SQL.
    if columns:
        available = {row[1] for row in conn.execute('PRAGMA table_info("' + table + '")')}
        chosen = [column for column in columns if column in available]
        if not chosen:
            return []
        query = "SELECT " + ",".join('"' + column + '"' for column in chosen) + ' FROM "' + table + '"'
    else:
        query = 'SELECT * FROM "' + table + '"'
    return [dict(row) for row in conn.execute(query)]


_HASHES = {}


def kernel_hash(name):
    if name not in _HASHES:
        _HASHES[name] = hashlib.sha256(name.encode("utf-8", "replace")).hexdigest()[:16]
    return _HASHES[name]


class GraphNodeLayers:
    """Map executed CUDA graph nodes to the Layer_N active when they were captured.

    Executed graph kernels carry the id of an instantiated (cloned) node; the
    clone points to the original node created during stream capture. Only a
    trace that recorded NVTX during that capture (nvtx-precapture, or capture
    inside the collection range) can resolve it; otherwise nothing is guessed.
    """

    def __init__(self, rows, regions):
        self.original = {}
        self.created = {}
        for row in rows:
            node = row.get("graphNodeId")
            if node is None:
                continue
            if row.get("originalGraphNodeId") is not None:
                self.original[node] = row["originalGraphNodeId"]
            if isinstance(row.get("start"), int) and row.get("globalTid") is not None:
                self.created.setdefault(node, (row["start"], row["globalTid"]))
        self.regions = regions
        self.cache = {}
        self.counts = collections.Counter()

    def resolve(self, node):
        if node in self.cache:
            return self.cache[node]
        seen = set()
        root = node
        while root in self.original and root not in seen:
            seen.add(root)
            root = self.original[root]
        created = self.created.get(root) or self.created.get(node)
        result = (None, None, "no_creation_event")
        if created is not None:
            layer, region = self.regions.find(created[1], created[0])
            if layer is not None or region is not None:
                result = (layer, region, "capture_time_nvtx")
            else:
                result = (None, None, "creation_outside_recorded_nvtx")
        self.cache[node] = result
        return result


def critical_chain(ops):
    """Walk back from the last-finishing op through latest-finishing predecessors.

    Returns ([(op, exclusive_ns)], gap_ns, envelope_ns). Each op contributes
    only the part after its predecessor finished, so PDL/stream overlap is not
    double counted, and contributions + gaps equal the envelope exactly.
    """
    items = sorted(ops, key=lambda op: (op["start"], op["end"]))
    if not items:
        return [], 0, 0
    starts = [op["start"] for op in items]
    best = []
    for index, op in enumerate(items):
        if not best or op["end"] > items[best[-1]]["end"]:
            best.append(index)
        else:
            best.append(best[-1])
    current = max(range(len(items)), key=lambda index: (items[index]["end"], items[index]["start"]))
    last_end = items[current]["end"]
    contributions = []
    gaps = 0
    while True:
        op = items[current]
        before = bisect.bisect_left(starts, op["start"]) - 1
        if before < 0:
            contributions.append((op, op["end"] - op["start"]))
            first_start = op["start"]
            break
        previous = items[best[before]]
        contributions.append((op, op["end"] - max(op["start"], previous["end"])))
        gaps += max(0, op["start"] - previous["end"])
        current = best[before]
    return contributions, gaps, last_end - first_start


def concurrency_shares(ops):
    """Split every elementary GPU interval equally among the ops active in it."""
    events = []
    for index, op in enumerate(ops):
        if op["end"] > op["start"]:
            events.append((op["start"], 1, index))
            events.append((op["end"], -1, index))
    events.sort(key=lambda event: (event[0], event[1]))
    shares = [0.0] * len(ops)
    active = set()
    previous = None
    for instant, delta, index in events:
        if previous is not None and active and instant > previous:
            part = (instant - previous) / len(active)
            for member in active:
                shares[member] += part
        if delta > 0:
            active.add(index)
        else:
            active.discard(index)
        previous = instant
    return shares


def add_ms(table, key, ns):
    table[key] = table.get(key, 0) + ns


def ranked(table, total, limit=None, name="key"):
    rows = [{name: key, "critical_ms": value / 1e6,
             "share_percent": 100 * value / total if total else None}
            for key, value in sorted(table.items(), key=lambda item: (-item[1], str(item[0])))]
    return rows[:limit] if limit else rows


def request_label_ok(label):
    return isinstance(label, str) and bool(REQUEST_LABEL_RE.match(label))


def private_directory_ok(path):
    """Kernel symbols may only be written into an owner-only directory."""
    parent = path.resolve().parent
    if not parent.is_dir():
        return False
    mode = parent.stat().st_mode
    return os.name == "posix" and stat.S_IMODE(mode) & 0o077 == 0 and not str(parent).startswith("/mnt/")


def regex_literal(text):
    # ECMAScript/ERE-safe literal escaping of only the metacharacters.
    return "".join("\\" + ch if ch in ".^$|?*+()[]{}\\" else ch for ch in text)


def shape_class(phase, position, count):
    """Position class of a forward inside its request (no M is inferred from timing)."""
    if phase == "decode":
        return "decode_M1"
    return "prefill_last_chunk" if position == count - 1 else "prefill_full_chunk"


def request_name(index, label):
    return label if label is not None else "request_%d" % index


def candidate_order(row):
    return (-row["critical_ms"], row["phase"], row["shape_class"], row["kernel_id"],
            str(row["grid"]), str(row["block"]), row["shared_memory_bytes"])


class ChainTable:
    """Critical-chain totals of one phase over a set of forward instances.

    The same instance is added to the all-measured table and to its request's
    table, so per-request and aggregate views use identical arithmetic.
    """

    def __init__(self):
        self.category, self.layer, self.layer_kind = {}, {}, {}
        self.layer_kind_category, self.region = {}, {}
        self.targets, self.target_meta = {}, {}
        self.inside_gaps = self.between_gaps = self.overlap_with_next = 0
        self.windows = self.instances = 0

    def add(self, chain, shape, regions, gap_to_next):
        self.instances += 1
        for op, ns in chain["contributions"]:
            add_ms(self.category, op["category"], ns)
            layer = op.get("layer")
            kind = regions.layer_kind(layer) if layer is not None else (
                "outside_layers" if op.get("region") else "unattributed")
            add_ms(self.layer, "unattributed" if layer is None else layer, ns)
            add_ms(self.layer_kind, kind, ns)
            add_ms(self.layer_kind_category, kind + "|" + op["category"], ns)
            add_ms(self.region, op.get("region") or "none", ns)
            if op["kind"] == "kernel":
                key = (shape, op["kernel_key"])
                add_ms(self.targets, key, ns)
                meta = self.target_meta.setdefault(key, {"op": op, "count": 0, "sum": 0, "layers": set()})
                meta["count"] += 1
                meta["sum"] += op["end"] - op["start"]
                if layer is not None:
                    meta["layers"].add(layer)
        self.inside_gaps += chain["gaps"]
        if gap_to_next is not None:
            if gap_to_next >= 0:
                self.between_gaps += gap_to_next
            else:
                self.overlap_with_next += -gap_to_next

    def total(self):
        return sum(self.category.values()) + self.inside_gaps + self.between_gaps

    def summary(self, regions):
        ops_total = sum(self.category.values())
        total = self.total()
        layer_rows = ranked(self.layer, total, name="layer")
        for layer_row in layer_rows:
            if isinstance(layer_row["layer"], int):
                layer_row["layer_kind"] = regions.layer_kind(layer_row["layer"])
        return {
            "instances_in_chain_total": self.instances, "request_windows": self.windows,
            "critical_total_ms": total / 1e6, "ops_critical_ms": ops_total / 1e6,
            "gpu_idle_inside_forward_ms": self.inside_gaps / 1e6,
            "gpu_idle_between_forwards_ms": self.between_gaps / 1e6,
            "forward_overlap_with_next_ms": self.overlap_with_next / 1e6,
            "by_category": ranked(self.category, total, name="category"),
            "by_layer_kind": ranked(self.layer_kind, total, name="layer_kind"),
            "by_layer_kind_category": ranked(self.layer_kind_category, total, 40, name="layer_kind_category"),
            "by_layer": layer_rows,
            "by_nvtx_region": ranked(self.region, total, name="region"),
        }

    def candidates(self, phase, request_total=None):
        phase_total = self.total()
        rows = []
        for (shape, key), ns in self.targets.items():
            meta = self.target_meta[(shape, key)]
            row = {
                "phase": phase, "shape_class": shape, "kernel_id": key[0], "category": meta["op"]["category"],
                "grid": key[1], "block": key[2], "shared_memory_bytes": key[3],
                "critical_ms": ns / 1e6,
                "share_of_phase_critical_percent": 100 * ns / phase_total if phase_total else None,
                "launches": meta["count"], "sum_duration_ms": meta["sum"] / 1e6,
                "mean_duration_us": meta["sum"] / meta["count"] / 1e3 if meta["count"] else None,
                "distinct_layers": len(meta["layers"]), "_key": key,
            }
            if request_total is not None:
                row["share_of_request_critical_percent"] = 100 * ns / request_total if request_total else None
            rows.append(row)
        return sorted(rows, key=candidate_order)


def public_rows(rows):
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def critical_path_section(ops, instances, regions, request_labels, run_receipt):
    """Critical chain per forward instance; aggregation per request and over all measured requests."""
    by_instance = collections.defaultdict(list)
    for op in ops:
        if op.get("instance") is not None:
            by_instance[op["instance"]].append(op)
    meta = {item["id"]: item for item in instances}
    ordered = [item for item in instances if by_instance.get(item["id"])]
    # Requests: one running request at a time, so a prefill instance after a
    # decode instance (or the first instance) opens the next request.
    requests = []
    previous_phase = None
    for item in ordered:
        if item["phase"] == "prefill" and previous_phase != "prefill":
            requests.append([])
        elif not requests:
            requests.append([])
        requests[-1].append(item)
        previous_phase = item["phase"]
    labels = list(request_labels or [])
    receipt_rows = []
    if run_receipt is not None:
        receipt_rows = [row for row in run_receipt.get("requests", []) if isinstance(row, dict)]
        labels = [row.get("label") for row in receipt_rows]
    for label in labels:
        if not request_label_ok(label):
            raise ValueError("request labels must match ^[a-z0-9_]{1,32}$")
    chains = {}
    for item in ordered:
        group = by_instance[item["id"]]
        contributions, gaps, envelope = critical_chain(group)
        first = min(op["start"] for op in group)
        last = max(op["end"] for op in group)
        chains[item["id"]] = {"contributions": contributions, "gaps": gaps, "envelope": envelope,
                              "first": first, "last": last, "busy": duration(spans(group)),
                              "cpu": item["end"] - item["start"]}
    aggregate = {phase: ChainTable() for phase in ("prefill", "decode")}
    request_rows = []
    per_request = []
    measured_instances = set()
    instance_request = {}
    request_names = {}
    # (phase, shape, kernel_key) -> request index -> [(op, critical ns > 0)], for
    # choosing the exact Compute instance of a candidate (shape and request aware).
    chain_ops = collections.defaultdict(lambda: collections.defaultdict(list))
    for index, members in enumerate(requests):
        label = labels[index] if index < len(labels) else None
        request_names[index] = request_name(index, label)
        # Without any labels every request counts; with labels, an unlabeled
        # extra segment (e.g. truncated tail) is never treated as measured.
        measured = (not labels) if label is None else not label.startswith(UNMEASURED_PREFIXES)
        row = {"index": index, "label": label, "measured": measured,
               "prefill_instances": sum(1 for m in members if m["phase"] == "prefill"),
               "decode_instances": sum(1 for m in members if m["phase"] == "decode")}
        prefill = [m for m in members if m["phase"] == "prefill"]
        decode = [m for m in members if m["phase"] == "decode"]
        for phase, group in (("prefill", prefill), ("decode", decode)):
            for position, member in enumerate(group):
                instance_request[member["id"]] = (index, label)
                shape = shape_class(phase, position, len(group))
                for op in by_instance[member["id"]]:
                    op["_request"], op["_shape"], op["_position"] = index, shape, position
        if prefill:
            first = chains[prefill[0]["id"]]["first"]
            last = max(chains[m["id"]]["last"] for m in prefill)
            row["prefill_gpu_window_ms"] = (last - first) / 1e6
            row["prefill_cpu_enqueue_ms"] = stats_ms([chains[m["id"]]["cpu"] for m in prefill])
            row["prefill_chunk_gpu_envelope_ms"] = stats_ms([chains[m["id"]]["envelope"] for m in prefill])
        periods = []
        between = []
        for left, right in zip(decode, decode[1:]):
            periods.append(chains[right["id"]]["first"] - chains[left["id"]]["first"])
            between.append(chains[right["id"]]["first"] - chains[left["id"]]["last"])
        if decode:
            row["decode_step_period_ms"] = stats_ms(periods)
            row["decode_step_gpu_envelope_ms"] = stats_ms([chains[m["id"]]["envelope"] for m in decode])
            row["decode_between_steps_gap_ms"] = stats_ms([max(0, g) for g in between])
            row["decode_cpu_enqueue_ms"] = stats_ms([chains[m["id"]]["cpu"] for m in decode])
        if index < len(receipt_rows):
            completion = receipt_rows[index].get("completion_tokens")
            if isinstance(completion, int) and completion > 0:
                row["receipt_completion_tokens"] = completion
                # The last prefill chunk samples the first token; each further
                # token is one decode forward (+1 possible overlap-scheduled step).
                row["decode_instances_minus_expected"] = row["decode_instances"] - (completion - 1)
                row["capture_complete"] = row["decode_instances"] >= completion - 1
        request_rows.append(row)
        if not measured:
            continue
        measured_instances.update(member["id"] for member in members)
        tables = {phase: ChainTable() for phase in ("prefill", "decode")}
        # Prefill window: chunk chains + positive gaps between chunks.
        for phase, group in (("prefill", prefill), ("decode", decode)):
            for position, member in enumerate(group):
                chain = chains[member["id"]]
                if phase == "decode" and position == len(group) - 1:
                    # Steady-state decode cost is a period; the last step of a
                    # request has no successor and is excluded from periods.
                    continue
                shape = shape_class(phase, position, len(group))
                gap = None
                if position + 1 < len(group):
                    gap = chains[group[position + 1]["id"]]["first"] - chain["last"]
                aggregate[phase].add(chain, shape, regions, gap)
                tables[phase].add(chain, shape, regions, gap)
                for op, ns in chain["contributions"]:
                    if op["kind"] == "kernel" and ns > 0:
                        chain_ops[(phase, shape, op["kernel_key"])][index].append((op, ns))
            if group:
                aggregate[phase].windows += 1
                tables[phase].windows += 1
        per_request.append((index, tables))
    phases = {phase: table.summary(regions) for phase, table in aggregate.items()}
    # Per-scenario view: warm short decode, fresh 16K and fresh 120K are never
    # mixed here; the all-measured phases above weight them by forward counts.
    by_request = []
    request_candidates = []
    for index, tables in per_request:
        name = request_names[index]
        request_total = sum(table.total() for table in tables.values())
        entry = {"index": index, "label": name, "request_critical_total_ms": request_total / 1e6,
                 "phases": {}}
        rows_in_request = []
        for phase, table in tables.items():
            summary = table.summary(regions)
            rows = table.candidates(phase, request_total)
            summary["ncu_candidates"] = public_rows(rows[:TOP_REQUEST_CANDIDATES])
            entry["phases"][phase] = summary
            rows_in_request.extend(rows)
        rows_in_request.sort(key=candidate_order)
        request_candidates.append((index, name, rows_in_request))
        by_request.append(entry)
    candidates = []
    for phase, table in aggregate.items():
        candidates.extend(table.candidates(phase))
    candidates.sort(key=candidate_order)
    for row in candidates:
        per = []
        for index, tables in per_request:
            ns = tables[row["phase"]].targets.get((row["shape_class"], row["_key"]), 0)
            if ns:
                per.append((ns, -index, request_names[index]))
        row["critical_ms_by_request"] = {name: ns / 1e6 for ns, _, name in sorted(per, key=lambda item: -item[1])}
        row["top_request_label"] = max(per)[2] if per else None
    return {"requests": request_rows, "phases": phases, "by_request": by_request,
            "candidates": public_rows(candidates[:TOP_CANDIDATES]), "all_candidates": candidates,
            "request_candidates": request_candidates, "chain_ops": chain_ops, "chains": chains,
            "measured_instances": measured_instances, "instance_request": instance_request,
            "request_names": request_names}


def graph_replay_section(ops):
    """Fallback without phase NVTX: each graph launch correlation is one replay."""
    replays = collections.defaultdict(list)
    for op in ops:
        if op.get("graph") and op.get("correlationId") is not None:
            replays[(op.get("globalPid"), op["correlationId"])].append(op)
    if not replays:
        return {"replay_count": 0}
    ordered = sorted(replays.values(), key=lambda group: min(op["start"] for op in group))
    totals = {}
    gaps = 0
    envelopes = []
    for group in ordered:
        contributions, gap, envelope = critical_chain(group)
        envelopes.append(envelope)
        gaps += gap
        for op, ns in contributions:
            add_ms(totals, op["category"], ns)
    periods = [min(op["start"] for op in b) - min(op["start"] for op in a) for a, b in zip(ordered, ordered[1:])]
    total = sum(totals.values()) + gaps
    return {"replay_count": len(ordered),
            "ops_per_replay": {"min": min(len(g) for g in ordered), "max": max(len(g) for g in ordered)},
            "replay_gpu_envelope_ms": stats_ms(envelopes), "replay_start_period_ms": stats_ms(periods),
            "chain_total_ms": total / 1e6, "gpu_idle_inside_replays_ms": gaps / 1e6,
            "by_category": ranked(totals, total, name="category"),
            "note": "Graph replays are not labelled decode without phase NVTX; envelope excludes host work between replays."}


def analyze(path, labels=None, request_labels=None, run_receipt=None, private_targets=None):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("input must be an existing SQLite file")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if TABLES["kernel"] not in tables:
            raise ValueError("Nsight kernel activity table is absent")
        strings = {row["id"]: row["value"] for row in read_rows(conn, "StringIds", tables)}
        data = {kind: read_rows(conn, table, tables, COLUMNS[kind]) for kind, table in TABLES.items()}
        enums = {row["id"]: row["name"] for row in read_rows(conn, "ENUM_CUDA_MEMCPY_OPER", tables)}
        graph_nodes = read_rows(conn, GRAPH_NODE_TABLE, tables, GRAPH_NODE_COLUMNS)
    if not enums:
        # CUPTI activity memcpy kinds; label fallback is recorded in the output.
        enums = {1: "CUDA_MEMCPY_KIND_HTOD", 2: "CUDA_MEMCPY_KIND_DTOH",
                 8: "CUDA_MEMCPY_KIND_DTOD", 9: "CUDA_MEMCPY_KIND_HTOH", 10: "CUDA_MEMCPY_KIND_PTOP"}
        copy_enum_source = "cupti_numeric_fallback"
    else:
        copy_enum_source = "sqlite_enum_table"
    invalid_counts = {}
    for kind, rows in data.items():
        valid = [row for row in rows if isinstance(row.get("start"), int)
                 and isinstance(row.get("end"), int) and row["end"] >= row["start"]]
        invalid_counts[kind] = len(rows) - len(valid)
        data[kind] = valid
    for row in data["nvtx"]:
        row["name"] = row.get("text") or strings.get(row.get("textId"), "")
    phase_labels = labels or PHASE_LABELS
    phases = PhaseIndex(data["nvtx"], phase_labels)
    instance_index = InstanceIndex(data["nvtx"], phase_labels)
    regions = RegionIndex(data["nvtx"])
    apis = data["runtime"] + data["driver"]
    for row in apis:
        row["name"] = strings.get(row.get("nameId"), "")
    correlations = CorrelationIndex(apis, phases)
    attribution = collections.Counter()
    for row in data["kernel"]:
        name = strings.get(row.get("demangledName"), strings.get(row.get("shortName"), ""))
        row["category"] = category(name)
        row["_name"] = name
    graph_layers = GraphNodeLayers(graph_nodes, regions)
    layer_methods = collections.Counter()
    ops = []
    for kind in ("kernel", "memcpy", "memset"):
        for row in data[kind]:
            row["phase"], reason = correlations.find(row)
            if kind == "kernel":
                attribution[reason] += 1
            candidates = correlations.candidates(row)
            instances = {instance_index.find(api) for api in candidates}
            instances.discard(None)
            instance = next(iter(instances)) if len(instances) == 1 else None
            layer = region = None
            method = "no_launch_correlation"
            if candidates:
                found = {regions.find(api["globalTid"], api["start"]) for api in candidates}
                if len(found) == 1:
                    layer, region = next(iter(found))
                    method = "launch_nvtx" if layer is not None or region is not None else "launch_outside_layer_nvtx"
                else:
                    method = "ambiguous_launch_nvtx"
            graph = bool(row.get("graphId") or row.get("graphNodeId"))
            if graph and row.get("graphNodeId") is not None and layer is None:
                node_layer, node_region, node_method = graph_layers.resolve(row["graphNodeId"])
                method = "graph_node_" + node_method
                if node_layer is not None or node_region is not None:
                    layer, region = node_layer, node_region or region
            layer_methods[(kind, row["phase"], method)] += 1
            if kind == "kernel":
                op_category = row["category"]
                shared = (row.get("staticSharedMemory") or 0) + (row.get("dynamicSharedMemory") or 0)
                launch_key = (kernel_hash(row["_name"]),
                              (row.get("gridX"), row.get("gridY"), row.get("gridZ")),
                              (row.get("blockX"), row.get("blockY"), row.get("blockZ")), shared)
            elif kind == "memcpy":
                op_category = "memcpy_" + copy_direction(row, enums)
                launch_key = None
            else:
                op_category = "memset"
                launch_key = None
            ops.append({"kind": kind, "start": row["start"], "end": row["end"],
                        "deviceId": row.get("deviceId"), "globalPid": row.get("globalPid"),
                        "correlationId": row.get("correlationId"), "phase": row["phase"],
                        "instance": instance, "category": op_category, "layer": layer,
                        "region": region, "graph": graph, "kernel_key": launch_key,
                        "_name": row.get("_name")})
    gpu = data["kernel"] + data["memcpy"] + data["memset"]
    devices = collections.defaultdict(list)
    for row in gpu:
        devices[row.get("deviceId")].append(row)
    device_summaries = []
    for index, (_, rows) in enumerate(sorted(devices.items(), key=lambda pair: str(pair[0]))):
        streams = collections.defaultdict(list)
        for row in rows:
            streams[(row.get("globalPid"), row.get("contextId"), row.get("streamId"))].append(row)
        stream_summaries = []
        for stream_index, stream_rows in enumerate(streams.values()):
            stream_summaries.append({"stream_index": stream_index, **timeline(stream_rows)})
        overlap, peak = simultaneous_streams(streams.values())
        kernels = [row for row in data["kernel"] if row.get("deviceId") == rows[0].get("deviceId")]
        copies = [row for row in data["memcpy"] if row.get("deviceId") == rows[0].get("deviceId")]
        device_summaries.append({"device_index": index, **timeline(rows),
                 "two_or_more_streams_overlap_ms": overlap / 1e6,
                 "peak_simultaneously_active_traced_streams": peak,
                 "kernel_memcpy_overlap_ms": intersection(spans(kernels), spans(copies)) / 1e6,
                 "streams": stream_summaries})
    copies = collections.defaultdict(list)
    for row in data["memcpy"]:
        copies[(copy_direction(row, enums), row["phase"])].append(row)
    copy_summary = []
    for (direction, phase), rows in sorted(copies.items()):
        copy_summary.append({"direction": direction, "phase": phase, "count": len(rows),
              "bytes": sum(row.get("bytes", 0) or 0 for row in rows),
              "sum_ms": sum(row["end"] - row["start"] for row in rows) / 1e6,
              "union_ms": duration(spans(rows)) / 1e6})
    api_groups = collections.defaultdict(list)
    # Runtime and driver time is kept separate: the two can be nested.
    for kind in ("runtime", "driver"):
        for row in data[kind]:
            api_groups[(kind, api_class(row["name"]))].append(row)
    api_summary = [{"source": kind, "category": label, "count": len(rows),
                    "inclusive_cpu_sum_ms": sum(r["end"] - r["start"] for r in rows) / 1e6,
                    "cpu_union_ms": duration(spans(rows)) / 1e6}
                   for (kind, label), rows in sorted(api_groups.items())]
    # Concurrency-share view: each elementary interval split among active ops;
    # per device the shares sum to the activity union, never to the kernel sum.
    share_table = collections.defaultdict(float)
    share_total = 0.0
    by_device = collections.defaultdict(list)
    for op in ops:
        by_device[op.get("deviceId")].append(op)
    for device_ops in by_device.values():
        for op, share in zip(device_ops, concurrency_shares(device_ops)):
            share_table[(op["phase"], op["category"])] += share
            share_total += share
    concurrency = [{"phase": phase, "category": cat, "wall_share_ms": value / 1e6,
                    "share_of_union_percent": 100 * value / share_total if share_total else None}
                   for (phase, cat), value in sorted(share_table.items(), key=lambda item: -item[1])]
    critical = critical_path_section(ops, instance_index.instances, regions, request_labels, run_receipt)
    graph_attributed = collections.Counter()
    for (kind, phase, method), count in layer_methods.items():
        graph_attributed[method] += count
    with path.open("rb") as source:
        sha = hashlib.file_digest(source, "sha256").hexdigest()
    result = {
        "schema_version": SCHEMA_VERSION, "input_sha256": sha, "cpu_only_analysis": True,
        "raw_strings_exported": False,
        "available_tables": {**{key: value in tables for key, value in TABLES.items()},
                             "graph_node_events": GRAPH_NODE_TABLE in tables},
        "invalid_or_incomplete_rows_excluded": invalid_counts,
        "phase_attribution": {"method": "same_thread_nvtx_contains_launch_api_then_cupti_correlation",
                              "global_tid_thread_bits": 24, "kernel_counts": dict(attribution),
                              "cross_thread_nvtx_ranges_ignored": phases.cross_thread_ranges_ignored},
        "phase_nvtx": phases.summary(),
        "all_kernels": kernel_summary(data["kernel"]),
        "phase_kernels": {phase: kernel_summary([r for r in data["kernel"] if r["phase"] == phase])
                          for phase in ("prefill", "decode", "unknown")},
        "graph_membership_kernels": {label: kernel_summary([r for r in data["kernel"]
                if bool(r.get("graphId") or r.get("graphNodeId")) == graph])
                for label, graph in (("graph", True), ("non_graph", False))},
        "devices": device_summaries, "memcpy_direction_source": copy_enum_source,
        "memory_copies": copy_summary if TABLES["memcpy"] in tables else None,
        "cuda_apis": api_summary,
        "forward_instances": {
            "count": len(instance_index.instances),
            "by_phase": dict(collections.Counter(item["phase"] for item in instance_index.instances)),
            "nested_phase_ranges_ignored": instance_index.nested_phase_ranges_ignored,
            "gpu_ops_with_instance": sum(1 for op in ops if op["instance"] is not None),
            "gpu_ops_without_instance": sum(1 for op in ops if op["instance"] is None),
        },
        "requests": critical["requests"],
        "critical_path": {
            "method": "per_forward_latest_finishing_predecessor_chain",
            "phases_scope": "all_measured_requests_mixed_by_forward_count",
            "phases": critical["phases"],
            "by_request": critical["by_request"],
            "graph_replays": graph_replay_section(ops),
        },
        "concurrency_wall_share": {"method": "elementary_interval_split_equally_among_active_gpu_ops",
                                   "total_ms": share_total / 1e6, "rows": concurrency},
        "sampler_nvtx": regions.sampler_summary(instance_index.instances),
        "layer_attribution": {
            "observed_layer_indices": len(regions.observed_layers),
            "mha_layers": sorted(layer for layer, value in regions.layer_has_mha.items() if value),
            "structural_label_counts": dict(regions.label_counts),
            "method_counts": [{"kind": kind, "phase": phase, "method": method, "count": count}
                              for (kind, phase, method), count in sorted(layer_methods.items())],
            "graph_node_events": len(graph_nodes),
        },
        "ncu_candidates": critical["candidates"],
        "interpretation_limits": [
            "Kernel shares divide SUM of kernel durations, not request latency or critical path.",
            "GPU attribution uses launch correlation, not overlap with CPU NVTX wall intervals.",
            "Graph membership alone does not identify decode. Unattributed work stays unknown.",
            "NVTX labels must exactly match configured fixed phase labels; no arbitrary text is exported.",
            "CPU nested NVTX/API sums can double count; union is separately reported.",
            "Device/stream timelines use interval unions; per-stream envelopes and gaps are not additive.",
            "Gaps only mean absence of recorded activity in this trace, not global GPU idle or CPU cost.",
            "CUPTI memcpy bytes are not PCIe totals: mapped-host gather kernels and UM paths can be omitted.",
            "Prefill/decode unions can overlap asynchronously and cannot be added as request wall time.",
            "Profiler capture boundaries may truncate queued activity; verify capture completeness externally.",
            "Profiling changes timing; speed acceptance requires separate unprofiled paired runs.",
            "Critical chain is a latest-finishing-predecessor walk inside each forward; it ignores explicit "
            "event dependencies and is exact only for serial streams (CUDA graph decode on one stream).",
            "Decode critical totals use step periods (first op to next step's first op); the last step of "
            "every request is excluded. Requests labelled warmup*/filler* are excluded from critical totals.",
            "Concurrency wall share splits overlapped time equally; it is a second view, not a dependency path.",
            "Decode graph layers are known only from capture-time NVTX (nvtx-precapture); otherwise "
            "layer stays unattributed and only categories are reported.",
            "ncu candidates are ranked by critical-chain time under the profiler; a Compute result on one "
            "kernel is a microbenchmark, not model speed.",
            "critical_path.phases mixes all measured requests (warm short decode, fresh 16K, fresh 120K) "
            "weighted by their forward counts; choose targets per scenario from critical_path.by_request.",
            "Shape classes are positions (decode step, full chunk, last chunk), not measured M; a launch "
            "config shared by several shapes or requests is flagged in the private targets.",
        ],
    }
    if private_targets is not None:
        write_private_targets(private_targets, critical, ops, sha)
        result["private_targets_written"] = True
    return result


def representative(entries):
    """Instance whose duration is closest to the median of the group (earliest on ties)."""
    median = percentile([op["end"] - op["start"] for op in entries], .5)
    return min(entries, key=lambda op: (abs((op["end"] - op["start"]) - median), op["start"]))


def ordinal_fields(op):
    return {"launch_skip_same_name_in_window": op["_ordinals"][0],
            "launch_skip_same_launch_config": op["_ordinals"][1],
            "launch_skip_same_name_same_phase": op["_ordinals"][2]}


def write_private_targets(target, critical, ops, sha):
    """Kernel symbols and ncu filters for the owner only (0600 in a 0700 dir).

    Each candidate gets one exact instance: same phase, shape class and full
    launch config (kernel, grid, block, shared memory) on the critical chain of
    the chosen request (the candidate's own request, or for all-measured
    candidates the request with the most critical time for that key). Every
    request/shape group that shares the launch config is listed with its own
    duration fingerprint and ordinals, so a Compute result can be checked
    against the intended M instead of trusting LaunchStats alone.
    """
    target = Path(target)
    if not private_directory_ok(target):
        raise PermissionError("private targets require an existing owner-only (0700) POSIX directory")
    names = critical["request_names"]
    chain_ops = critical["chain_ops"]
    ordered = sorted((op for op in ops if op["kind"] == "kernel"), key=lambda op: (op["start"], op["end"]))
    by_name = collections.defaultdict(int)
    by_config = collections.defaultdict(int)
    by_phase = collections.defaultdict(int)
    first = {}
    for op in ordered:
        name = op["_name"]
        key = op["kernel_key"]
        if key[0] not in first:
            first[key[0]] = name
        ordinal = (by_name[name], by_config[key], by_phase[(name, op["phase"])])
        op["_ordinals"] = ordinal
        by_name[name] += 1
        by_config[key] += 1
        by_phase[(name, op["phase"])] += 1
    by_launch_key = collections.defaultdict(list)
    for op in ordered:
        by_launch_key[op["kernel_key"]].append(op)
    group_cache = {}

    def launch_config_groups(key):
        """All request/shape groups of this exact launch config in the window."""
        if key not in group_cache:
            groups = collections.OrderedDict()
            for op in by_launch_key[key]:
                request = op.get("_request")
                group = (request, op.get("_shape") or "outside_requests")
                groups.setdefault(group, []).append(op)
            rows = []
            for (request, shape), members in sorted(groups.items(), key=lambda item: (
                    -1 if item[0][0] is None else item[0][0], item[0][1])):
                durations = [op["end"] - op["start"] for op in members]
                rows.append({"request_index": request,
                             "request_label": None if request is None else names.get(request),
                             "shape_class": shape, "launches": len(members),
                             "p50_duration_us": percentile(durations, .5) / 1e3,
                             "min_duration_us": min(durations) / 1e3, "max_duration_us": max(durations) / 1e3,
                             **ordinal_fields(representative(members))})
            group_cache[key] = rows
        return group_cache[key]

    def target_row(candidate, scope, rank, request_index):
        key = candidate["_key"]
        entries = [op for op, _ in chain_ops[(candidate["phase"], candidate["shape_class"], key)].get(request_index, [])]
        name = first.get(key[0], "")
        row = {**{k: v for k, v in candidate.items() if not k.startswith("_")},
               "scope": scope, "rank": rank, "demangled_name": name,
               "ncu_kernel_regex": "regex:^" + regex_literal(name) + "$"}
        if not entries:
            row.update(instance_request_index=None, instance_request_label=None)
            return row
        chosen = representative(entries)
        durations = [op["end"] - op["start"] for op in entries]
        groups = [dict(group, selected=group["request_index"] == request_index
                       and group["shape_class"] == candidate["shape_class"])
                  for group in launch_config_groups(key)]
        row.update({
            "instance_request_index": request_index, "instance_request_label": names.get(request_index),
            "instance_shape_class": chosen.get("_shape"), "instance_chunk_or_step_position": chosen.get("_position"),
            "instance_layer": chosen.get("layer"), "instance_duration_us": (chosen["end"] - chosen["start"]) / 1e3,
            "chain_instances_in_request": len(entries),
            "chain_duration_p50_us_in_request": percentile(durations, .5) / 1e3,
            **ordinal_fields(chosen),
            "launch_config_identifies_instance_group": len(groups) == 1,
            "other_shape_classes_same_launch_config": sorted({group["shape_class"] for group in groups
                                                              if group["shape_class"] != candidate["shape_class"]}),
            "launch_config_groups": groups,
        })
        return row

    rows = []
    for rank, candidate in enumerate(critical["all_candidates"][:TOP_CANDIDATES * 2], 1):
        per_request = chain_ops[(candidate["phase"], candidate["shape_class"], candidate["_key"])]
        # Request carrying the most critical time for this exact key (earliest on ties).
        best = max(per_request, key=lambda index: (sum(ns for _, ns in per_request[index]), -index)) \
            if per_request else None
        rows.append(target_row(candidate, "all_measured", rank, best))
    for index, name, candidates in critical["request_candidates"]:
        for rank, candidate in enumerate(candidates[:TOP_REQUEST_CANDIDATES * 2], 1):
            rows.append(target_row(candidate, "request:" + name, rank, index))
    payload = {"schema_version": SCHEMA_VERSION, "input_sha256": sha, "private": True,
               "note": "Ordinals count kernels from the first recorded GPU op of this capture; ncu with "
                       "--profile-from-start off counts from the same cudaProfilerStart. The instance is "
                       "matched by phase, shape class and full launch config inside instance_request; "
                       "launch_config_groups lists every group sharing that config with its duration "
                       "fingerprint, so check the Compute result against the intended group.",
               "candidates": rows}
    fd, temporary = tempfile.mkstemp(prefix=".nsys-targets-", dir=target.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, indent=2, allow_nan=False, default=list)
            output.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-label", action="append", default=[])
    parser.add_argument("--decode-label", action="append", default=[])
    parser.add_argument("--request-labels", default=None,
                        help="comma-separated labels for requests in order, e.g. warmup_short,warm_decode")
    parser.add_argument("--run-receipt", type=Path, default=None,
                        help="profile-request.py run receipt; labels and completion counts are taken from it")
    parser.add_argument("--private-targets", type=Path, default=None,
                        help="owner-only JSON with kernel symbols and ncu filters (0700 directory required)")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    if args.private_targets is not None and args.private_targets.resolve() in (args.input.resolve(), args.output.resolve()):
        parser.error("private targets must differ from input and output")
    labels = {phase: names | set(getattr(args, phase + "_label")) for phase, names in PHASE_LABELS.items()}
    if labels["prefill"] & labels["decode"]:
        parser.error("phase labels must be disjoint")
    if args.request_labels and args.run_receipt:
        parser.error("use either --request-labels or --run-receipt")
    request_labels = None
    if args.request_labels:
        request_labels = [item.strip() for item in args.request_labels.split(",") if item.strip()]
        if not all(request_label_ok(item) for item in request_labels):
            parser.error("request labels must match ^[a-z0-9_]{1,32}$")
    receipt = None
    if args.run_receipt:
        receipt = json.loads(args.run_receipt.read_text(encoding="utf-8"))
    result = analyze(args.input, labels, request_labels, receipt, args.private_targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Write a complete numeric/sanitized artifact atomically; no raw trace copy.
    fd, temporary = tempfile.mkstemp(prefix=".nsys-summary-", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            json.dump(result, output, indent=2, allow_nan=False, default=list)
            output.write("\n")
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"kernel_count": result["all_kernels"]["kernel_count"],
                      "sum_kernel_ms": result["all_kernels"]["sum_kernel_ms"],
                      "phase_nvtx_counts": {k: v["range_count"] for k, v in result["phase_nvtx"].items()},
                      "forward_instances": result["forward_instances"]["count"],
                      "requests": len(result["requests"]),
                      "output_written": True}))


if __name__ == "__main__":
    main()
