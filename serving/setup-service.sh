#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/USER/qwen36-api
DELIVERABLES=/path/to/qwen36-workspace/outputs/qwen36-api
mkdir -p "$ROOT/secrets" "$ROOT/config" "$HOME/.config/systemd/user"
chmod 700 "$ROOT" "$ROOT/secrets" "$ROOT/config"
test -f "$ROOT/config/runtime.env" || printf 'QWEN36_STRATEGY=offload\n' > "$ROOT/config/runtime.env"
install -m 700 "$DELIVERABLES/guard.py" "$DELIVERABLES/placement.py" "$DELIVERABLES/request_metrics.py" "$DELIVERABLES/metrics_summary.py" "$DELIVERABLES/serve_guarded.py" "$DELIVERABLES/start-inference.sh" "$DELIVERABLES/readiness.py" "$ROOT/"
install -m 600 "$DELIVERABLES/qwen36-inference.service" "$HOME/.config/systemd/user/"
"$ROOT/.venv/bin/python" - <<'PY'
import pathlib,secrets
p=pathlib.Path('/home/USER/qwen36-api/secrets/api.env')
if not p.exists(): p.write_text('QWEN36_API_KEY='+secrets.token_urlsafe(32)+'\n')
p.chmod(0o600)
PY
systemctl --user daemon-reload
"$ROOT/.venv/bin/python" -m py_compile "$ROOT/guard.py" "$ROOT/serve_guarded.py"
cp "$ROOT/requirements.lock.txt" "$ROOT/serve-help.txt" "$ROOT/bench-help.txt" "$DELIVERABLES/"
stat -c '%a %n' "$ROOT/secrets/api.env"
