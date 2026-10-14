"""Fallback / provenance scenarios of the integrated GEMV, one per process (GPU, under the lock).

Real layers: 0 (GDN in_proj_qkvz 12288x2048) and 3 (attention qkv_proj 9216x2048), plus layer 0's
out_proj (uniform, cuBLASLt). Prints one JSON line {"case", "pass", ...}.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LIB, W, Checkpoint, GemvCounter, activation, load_orig_fpl, load_projections, make_layer, mismatches  # noqa: E402

import torch  # noqa: E402

# ABI-v1 build of the earlier analysis library (must be refused by the provenance check)
V1_LIB = os.environ.get("QWEN36_FP8_GEMV_V1_LIBRARY", "/path/to/v1/libqwen_fp8_gemv.so")


def memfd_mapped() -> bool:
    with open("/proc/self/maps") as f:
        return "memfd:qwen_fp8_gemv" in f.read()


def compare_all(kern, orig, projs, layer_key, ms=range(1, 9)):
    """Mismatch count of kern.apply(layer) vs pristine over M=1..8 for every projection."""
    dev = torch.device("cuda", 0)
    bad = torch.zeros(1, dtype=torch.int64, device=dev)
    gen = torch.Generator().manual_seed(31337)
    for M in ms:
        for p in projs:
            x = activation("normal", M, p["K"], p["xs_value"], gen)
            pristine = orig.fp8_pertensor_linear(x, p["W"], p["S"], None, p["xs"], p[layer_key]._fp8_uniform_scale,
                                                 scale_segments=p[layer_key]._fp8_scale_segments)
            bad[0] += mismatches(kern.apply(p[layer_key], x), pristine)
    return int(bad.item())


def main():
    case = sys.argv[1]
    os.environ.pop("QWEN36_FP8_GEMV", None)
    os.environ.pop("QWEN36_FP8_GEMV_LIBRARY", None)
    torch.cuda.set_device(0)
    side = torch.cuda.Stream()
    torch.cuda.set_stream(side)
    orig = load_orig_fpl()
    from freetoken.layers.quantization.linear.fp8_tensor import TorchFp8TensorLinearKernel

    kern = TorchFp8TensorLinearKernel()
    ck = Checkpoint()
    projs = load_projections(ck, layers=[0, 3], kinds=("in",)) + load_projections(ck, layers=[0], kinds=("out",))
    for p in projs:
        p["xs_value"] = float(p["xs"].item())
    res = {"case": case}
    tmp = tempfile.mkdtemp(dir=str(W / "build"))
    try:
        if case == "env_unset":
            for p in projs:
                p["L"] = make_layer(p["W"], p["S"], p["xs"])
                kern.finalize(p["L"])
            res["mismatches"] = compare_all(kern, orig, projs, "L")
            res["module_imported"] = "freetoken.kernel.cuda_fp8_gemv" in sys.modules
            res["library_mapped"] = memfd_mapped()
            res["pass"] = res["mismatches"] == 0 and not res["module_imported"] and not res["library_mapped"]
        else:
            os.environ["QWEN36_FP8_GEMV"] = "1"
            os.environ["QWEN36_FP8_GEMV_LIBRARY"] = str(LIB)
            import freetoken.kernel.cuda_fp8_gemv as g

            counter = None
            if case == "selftest_mismatch_m3_m5":
                real_ref = g._reference

                def bad_ref(qa, weight, input_scale, weight_scale):
                    out = real_ref(qa, weight, input_scale, weight_scale)
                    if qa.shape[0] in (3, 5):  # pretend torch picked another CUTLASS config for these M
                        out = out.clone()
                        out.view(torch.int16)[0, 0] ^= 1
                    return out

                g._reference = bad_ref
            elif case == "selftest_launch_error":
                g.VARIANT = 99  # the library rejects it: every self-test launch returns an error
            elif case == "platform_pin":
                g.TORCH_PIN = (2, 99)
            elif case == "provenance_tampered":
                bad = os.path.join(tmp, "libqwen_fp8_gemv.so")
                with open(LIB, "rb") as src, open(bad, "wb") as dst:
                    dst.write(src.read() + b"\0")
                os.environ["QWEN36_FP8_GEMV_LIBRARY"] = bad
            elif case == "provenance_stale_v1":
                os.environ["QWEN36_FP8_GEMV_LIBRARY"] = V1_LIB
            elif case == "provenance_missing_path":
                os.environ.pop("QWEN36_FP8_GEMV_LIBRARY")
            elif case == "real_weight_mismatch_m2":
                real_ref = g._reference
                target = projs[1]["W"].data_ptr()  # layer 3 qkv_proj real weight

                def bad_ref2(qa, weight, input_scale, weight_scale):
                    out = real_ref(qa, weight, input_scale, weight_scale)
                    if qa.shape[0] == 2 and weight.data_ptr() == target:
                        out = out.clone()
                        out.view(torch.int16)[0, 0] ^= 1
                    return out

                g._reference = bad_ref2
            elif case == "capture_guard":
                # prepare_layer reached under capture must neither sync nor decide (the capture
                # would be invalidated); the eager finalize afterwards decides normally
                p = projs[0]
                gr = torch.cuda.CUDAGraph()
                ones = torch.ones(4, device="cuda")
                with torch.cuda.graph(gr, stream=side):
                    res["prepare_under_capture"] = g.prepare_layer(make_layer(p["W"], p["S"], p["xs"]))
                    _ = ones * 2
                res["decided_under_capture"] = dict(g._DECIDED)
                res["state_under_capture"] = g._STATE is not None
            raised = None
            if case == "threads_first_decision":
                calls = []
                real_self_test = g._self_test

                def counted(*a, **k):
                    calls.append(1)
                    return real_self_test(*a, **k)

                g._self_test = counted
                layers = [make_layer(projs[i % 2]["W"], projs[i % 2]["S"], projs[i % 2]["xs"]) for i in range(4)]
                errs = []

                def fin(layer):
                    try:
                        kern.finalize(layer)
                    except Exception as exc:  # noqa: BLE001
                        errs.append(repr(exc))

                ths = [threading.Thread(target=fin, args=(L,)) for L in layers]
                for t in ths:
                    t.start()
                for t in ths:
                    t.join()
                res.update(self_test_runs=len(calls), flags=[L._fp8_gemv for L in layers], errors=errs)
                res["pass"] = len(calls) == 1 and all(L._fp8_gemv for L in layers) and not errs
            else:
                try:
                    for p in projs:
                        p["L"] = make_layer(p["W"], p["S"], p["xs"])
                        kern.finalize(p["L"])
                except Exception as exc:  # noqa: BLE001
                    raised = exc
                res["raised"] = None if raised is None else f"{type(raised).__name__}: {raised}"[:300]
                res["status"] = g.status()
                res["library_mapped"] = memfd_mapped()
                if raised is None:
                    counter = GemvCounter()
                    if case == "runtime_launch_error":
                        res["enabled_before"] = g.status()["enabled"]
                        g.VARIANT = 99  # first runtime launch fails -> disable for good, CUTLASS output
                    res["flags"] = [p["L"]._fp8_gemv for p in projs]
                    res["mismatches"] = compare_all(kern, orig, projs, "L")
                    res["served"] = {f"{n}xM{m}": v for (n, m), v in sorted(counter.served.items())}
                    res["status_after"] = g.status()
                st = res["status"]
                served_m = sorted({int(k.split("xM")[1]) for k in res.get("served", {})})
                if case.startswith("provenance_"):
                    res["pass"] = (raised is not None and type(raised).__name__ == "Fp8GemvProvenanceError"
                                   and not res["library_mapped"])
                elif case == "selftest_mismatch_m3_m5":
                    res["pass"] = (raised is None and res["mismatches"] == 0
                                   and st.get("allowed_m") == {"12288x2048": [1, 2, 4, 6, 7, 8],
                                                               "9216x2048": [1, 2, 4, 6, 7, 8]}
                                   and served_m == [1, 2, 4, 6, 7, 8])
                elif case in ("selftest_launch_error", "platform_pin"):
                    res["pass"] = (raised is None and res["mismatches"] == 0 and not st["enabled"]
                                   and not any(res["flags"]) and not res["served"]
                                   and (case != "platform_pin" or not res["library_mapped"]))
                elif case == "runtime_launch_error":
                    res["pass"] = (raised is None and res["enabled_before"] and res["mismatches"] == 0
                                   and not res["served"] and not res["status_after"]["enabled"])
                elif case == "real_weight_mismatch_m2":
                    # the failing (shape, M) leaves the GEMV for every layer of that shape only
                    res["pass"] = (raised is None and res["mismatches"] == 0 and st["enabled"]
                                   and st.get("allowed_m", {}).get("9216x2048") == [1, 3, 4, 5, 6, 7, 8]
                                   and st.get("allowed_m", {}).get("12288x2048") == [1, 2, 3, 4, 5, 6, 7, 8]
                                   and "9216xM2" not in res["served"] and "12288xM2" in res["served"]
                                   and len(st["layer_mismatch_events"]) == 1)
                elif case == "capture_guard":
                    res["pass"] = (res.get("prepare_under_capture") is False and res.get("decided_under_capture") == {}
                                   and res.get("state_under_capture") is False
                                   and raised is None and st["enabled"] and res["mismatches"] == 0)
                else:
                    res["pass"] = False
                    res["error"] = "unknown case"
    except Exception as exc:  # noqa: BLE001
        res["pass"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(res, default=str), flush=True)
    return 0 if res.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
