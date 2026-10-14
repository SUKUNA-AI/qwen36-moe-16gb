#!/usr/bin/env bash
set -euo pipefail
workspace='/path/to/qwen36-workspace'
export PYTHONPATH="$workspace/work/FreeToken-opt/python:$workspace/outputs/qwen36-api"
python=/home/USER/qwen36-api/.venv/bin/python
# Guard under test for the tool_choice/v2 suites; a native run passes the service-root guard.py.
guard="${QWEN36_GUARD_UNDER_TEST:-$workspace/outputs/qwen36-api/guard.py}"
"$python" "$workspace/outputs/qwen36-api/test-guard.py"
CUDA_VISIBLE_DEVICES= "$python" "$workspace/outputs/qwen36-api/optimization/test-tool-choice-guard-cpu.py" --guard "$guard"
CUDA_VISIBLE_DEVICES= "$python" "$workspace/outputs/qwen36-api/optimization/test-tool-semantics-guard-cpu.py" --guard "$guard" --gate-script "${BASH_SOURCE[0]}"
"$python" "$workspace/outputs/qwen36-api/test-metrics.py"
"$python" "$workspace/outputs/qwen36-api/optimization/test-log-failure.py"
"$python" - "$workspace" <<'PY'
import pathlib,runpy,sys
tests=runpy.run_path(str(pathlib.Path(sys.argv[1])/'work/FreeToken-opt/tests/engine/test_speculative.py'))
for name,fn in tests.items():
    if name.startswith('test_'):fn();print(name,'PASS')
PY
