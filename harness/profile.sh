#!/usr/bin/env bash
# Nsight launcher for one experiment unit (started by start-experiment.sh).
# Legacy fixtures keep their nsys options. profile-final-* configs carry an
# "nsys" block (tool nsys or ncu) validated below; every report gets a unique
# timestamped name (existing evidence is never overwritten) and mode 0600.
set -euo pipefail
umask 077
root=/home/USER/qwen36-api/optimization
py=/home/USER/qwen36-api/.venv/bin/python
nsys="$root/tools/target-linux-x64/nsys"
ncu="$root/tools/ncu/ncu"
reports="$root/reports"
profile="${1:-profile64}"
case "$profile" in
  profile64|profile-cuda128|profile-main128) ;;
  profile-final-*)
    if ! [[ "$profile" =~ ^profile-final-[a-z0-9][a-z0-9-]{0,63}$ ]]; then
      echo 'Bad final profile name' >&2; exit 64
    fi ;;
  *) echo 'Unknown profiler fixture' >&2; exit 64;;
esac
config="$root/config/$profile.json"
if [ ! -f "$config" ]; then echo "Missing profile config $profile" >&2; exit 66; fi
mkdir -p "$reports"
chmod 700 "$reports"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
spec=$(mktemp "$reports/.profile-spec-XXXXXX")
trap 'rm -f "$spec"' EXIT

# The helper prints NUL-separated: tool, output prefix, then tool arguments.
# Values never pass through a shell parser.
"$py" -I - "$config" "$profile" "$root" > "$spec" <<'PY'
import json, re, sys
config_path, profile, root = sys.argv[1:4]
config = json.load(open(config_path, encoding="utf-8"))
def fail(message):
    sys.stderr.write("profile config rejected: " + message + "\n"); sys.exit(65)
env = config.get("env", {})
if not isinstance(env, dict) or not str(env.get("QWEN36_NSYS_FORWARDS", "")).isdigit() \
        or not 0 < int(env["QWEN36_NSYS_FORWARDS"]) < 100000:
    fail("env.QWEN36_NSYS_FORWARDS must be a positive integer (cudaProfilerStart/Stop window)")
legacy = {"profile64": "model-baseline", "profile-cuda128": "model-cuda128", "profile-main128": "model-main128"}
out = []
if profile in legacy:
    out = ["nsys", legacy[profile], "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--cuda-graph-trace=node"]
else:
    block = config.get("nsys")
    if not isinstance(block, dict):
        fail("profile-final config needs an nsys block")
    prefix = block.get("output_prefix", "")
    if not re.fullmatch(r"(model|ncu)-final-[a-z0-9][a-z0-9-]{0,80}", prefix or ""):
        fail("output_prefix must look like model-final-<name> or ncu-final-<name>")
    tool = block.get("tool", "nsys")
    if tool == "nsys":
        graph = block.get("graph_nvtx", "precapture")
        if graph not in ("precapture", "live"):
            fail("graph_nvtx must be precapture or live")
        event = block.get("cuda_event_trace", "false")
        if event not in ("false", "auto"):
            fail("cuda_event_trace must be false or auto")
        graph_trace = "--cuda-graph-trace=node:nvtx-precapture" if graph == "precapture" else "--cuda-graph-trace=node"
        # OSRT stays off: its fork helper injection hung /usr/bin/file on this WSL build.
        out = ["nsys", prefix, "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop-shutdown", "--kill=sigterm",
               graph_trace, "--cuda-event-trace=" + event, "--wait=primary", "--export=sqlite"]
    elif tool == "ncu":
        spec = config.get("ncu")
        if not isinstance(spec, dict):
            fail("tool ncu needs an ncu block")
        regex = spec.get("kernel_regex", "")
        if not isinstance(regex, str) or not regex.startswith("regex:") or not 7 < len(regex) <= 4096 \
                or any(ch in regex for ch in "\x00\n\r"):
            fail("ncu.kernel_regex must be regex:<expression> without control characters")
        allowed_sections = {"SpeedOfLight", "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Chart",
                            "MemoryWorkloadAnalysis_Tables", "ComputeWorkloadAnalysis", "Occupancy",
                            "WarpStateStats", "SourceCounters", "SchedulerStats", "LaunchStats",
                            "InstructionStats", "SpeedOfLight_RooflineChart", "WorkloadDistribution"}
        sections = spec.get("sections", [])
        if not sections or not all(s in allowed_sections for s in sections):
            fail("ncu.sections must be a non-empty subset of the installed section identifiers")
        skip, count = spec.get("launch_skip", 0), spec.get("launch_count", 1)
        if not (isinstance(skip, int) and 0 <= skip < 10**8 and isinstance(count, int) and 1 <= count <= 32):
            fail("ncu.launch_skip/launch_count out of range")
        mode = spec.get("filter_mode", "global")
        phase = spec.get("phase", "any")
        cache = spec.get("cache_control", "all")
        clock = spec.get("clock_control", "base")
        base = spec.get("kernel_name_base", "demangled")
        if mode not in ("global", "per-launch-config") or phase not in ("decode", "prefill", "any") \
                or cache not in ("all", "none") or clock not in ("base", "none", "boost") \
                or base not in ("demangled", "function", "mangled"):
            fail("ncu filter/phase/cache/clock/name-base value not allowed")
        if not str(env.get("QWEN36_STALL_SECONDS", "")).isdigit() or int(env["QWEN36_STALL_SECONDS"]) < 900:
            fail("ncu configs need env.QWEN36_STALL_SECONDS >= 900: kernel replay stalls forwards")
        metrics = spec.get("metrics", [])
        if not isinstance(metrics, list) or len(metrics) > 16 or not all(
                isinstance(m, str) and re.fullmatch(r"[a-z][a-z0-9_]*__[a-z0-9_]+(\.[a-z0-9_]+)?", m) for m in metrics):
            fail("ncu.metrics must be at most 16 metric names like smsp__inst_executed.sum")
        out = ["ncu", prefix, "--target-processes", "all", "--target-processes-filter", "exclude:file",
               "--profile-from-start", "off", "--kill", "1", "--replay-mode", "kernel",
               "--graph-profiling", "node", "--kernel-name-base", base, "--kernel-name", regex,
               "--filter-mode", mode, "--launch-skip", str(skip), "--launch-count", str(count),
               "--cache-control", cache, "--clock-control", clock, "--import-source", "no",
               "--section-folder", root + "/sections"]
        if phase != "any":
            # ncu push/pop syntax: '/' inside a range name is escaped, trailing '/' marks push/pop.
            out += ["--nvtx", "--nvtx-include", "Qwen36\\/" + phase + "/"]
        for section in sections:
            out += ["--section", section]
        if metrics:
            # Clock/cache independent counters (instructions, DRAM bytes) for the M check.
            out += ["--metrics", ",".join(metrics)]
    else:
        fail("nsys.tool must be nsys or ncu")
sys.stdout.write("\0".join(out) + "\0")
PY

mapfile -d '' -t fields < "$spec"
tool="${fields[0]}"
prefix="${fields[1]}"
args=("${fields[@]:2}")
base="$prefix-$stamp"
n=1
while compgen -G "$reports/$base.*" > /dev/null; do n=$((n + 1)); base="$prefix-$stamp-$n"; done
output="$reports/$base"
# Pointer for profile-request.py: which report this unit invocation is producing.
# INVOCATION_ID (set by systemd for this unit run) lets the client reject a stale pointer.
"$py" -I -c 'import json,sys; json.dump({"profile":sys.argv[1],"tool":sys.argv[2],"output_base":sys.argv[3],"started_utc":sys.argv[4],"invocation_id":sys.argv[5]},open(sys.argv[6],"w"))' \
  "$profile" "$tool" "$base" "$started" "${INVOCATION_ID:-}" "$reports/profile-final-current.json"
chmod 600 "$reports/profile-final-current.json"

status=0
if [ "$tool" = nsys ]; then
  "$nsys" profile "${args[@]}" --force-overwrite=false --output="$output" \
    /bin/bash "$root/run-experiment.sh" "$profile" || status=$?
  version=$("$nsys" --version 2>/dev/null | head -1 || true)
else
  "$ncu" "${args[@]}" -o "$output" /bin/bash "$root/run-experiment.sh" "$profile" || status=$?
  version=$("$ncu" --version 2>/dev/null | tail -1 || true)
  for report in "$output.ncu-rep" "$output.ncu-repz"; do
    if [ -f "$report" ]; then
      "$ncu" --import "$report" --csv --page details --print-units base > "$output.details.csv" || true
      "$ncu" --import "$report" --csv --page raw --print-units base > "$output.raw.csv" || true
    fi
  done
fi
find "$reports" -maxdepth 1 -type f -name "$base*" -exec chmod 600 {} +
"$py" -I - "$config" "$profile" "$tool" "$base" "$reports" "$started" "$status" "$version" "${args[@]}" <<'PY'
import datetime, hashlib, json, os, sys
config, profile, tool, base, reports, started, status, version = sys.argv[1:9]
outputs = []
for name in sorted(os.listdir(reports)):
    if name.startswith(base + ".") and not name.endswith(".profile-receipt.json"):
        path = os.path.join(reports, name)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        outputs.append({"name": name, "bytes": os.path.getsize(path), "sha256": digest.hexdigest()})
receipt = {"schema": 1, "profile": profile, "tool": tool, "tool_version": version,
           "config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest(),
           "tool_arguments": sys.argv[9:], "started_utc": started,
           "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "exit_status": int(status), "outputs": outputs,
           "note": "Profiler timings are diagnostic only; speed acceptance needs unprofiled paired runs."}
path = os.path.join(reports, base + ".profile-receipt.json")
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as handle:
    json.dump(receipt, handle, indent=2)
PY
exit "$status"
