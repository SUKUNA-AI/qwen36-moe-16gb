"""eval-v1 suite for the FP8-KV evaluation (wrapper; the frozen eval-v1 sources are imported unchanged).

The original eval-v1.py pins two basis profiles (bf128, bf128-fp16compat). This wrapper reuses the same
60 cases, runner, sandbox and receipts, and only replaces:
  - PROFILES: main-release and main-release-fp8kv (read from ./configs, byte copies of the run configs);
  - FREEZE:   ./eval-fp8kv-freeze.json (new freeze, O_EXCL; the case list and suite sha256 are unchanged);
  - the decision rule recorded in the freeze: RULE of ./eval-fp8kv-compare.py.

Modes (as eval-v1): verify | freeze --out PATH [--tokenizer DIR] | run --name N --expect-profile P.
The 'managed' mode is NOT used (it requires workspace configs under outputs/); the GPU driver
gpu/s2-eval.sh starts/stops the unit itself under the GPU lock and calls 'run'.
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OPT = pathlib.Path('/path/to/qwen36-workspace/outputs/qwen36-api/optimization')
sys.path.insert(0, str(OPT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load('eval_v1', OPT / 'eval-v1.py')
rule_module = _load('eval_fp8kv_compare', HERE / 'eval-fp8kv-compare.py')

ev.FREEZE = HERE / 'eval-fp8kv-freeze.json'
ev.PROFILES = {name: json.loads((HERE / 'configs' / f'{name}.json').read_text())
               for name in ('main-release', 'main-release-fp8kv')}
ev.load_compare_module = lambda: rule_module

if __name__ == '__main__':
    raise SystemExit(ev.main())
