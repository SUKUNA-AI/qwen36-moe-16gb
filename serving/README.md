# serving/ — runtime, release-managed files and operations

Files the release installs and the operator uses to run Qwen3.6-35B-A3B-NVFP4 behind the
guarded OpenAI-compatible API (Linux side in WSL2, Windows supervisor on the host).

| Path | Role |
|---|---|
| `profile-runtime.py`, `profiles.json` | Profile schema/validation and launch mapping (`main128`, `long256`, `baseline64`); every switch maps to a `QWEN36_*` env of the fork. |
| `start-inference.sh`, `qwen36-inference.service` | systemd user unit and launcher (exports `GLIBC_TUNABLES`; `profile-runtime.py` builds the clean `QWEN36_*`/`FREETOKEN_*` environment). |
| `serve_guarded.py`, `guard.py`, `ingress.py` | API guard: key check, request validation, tool-choice semantics, rate/size limits. |
| `request_metrics.py`, `metrics_summary.py`, `placement.py` | Bounded asynchronous request metrics and summaries. |
| `readiness.py`, `keepalive.py`, `wait-ready.py`, `smoke.py` | Readiness (local and through the edge), WSL keepalive, smoke checks. |
| `supervisor.ps1`, `watch-supervisor.ps1`, `manage.ps1`, `bin/qwen36.cmd` | Windows supervisor (model unit, keepalive, SSH reverse tunnel) and the `qwen36` command. |
| `get-api-key.ps1`, `configure-manual-command.ps1`, `host-metrics.ps1` | Operator helpers (the key is read from the secrets directory at run time, never stored here). |
| `install-cuda-13.4.sh`, `check-toolchain-13.4.sh`, `cuda-13.4.1-manifest.json`, `install-runtime.sh`, `setup-service.sh`, `requirements.lock.txt`, `versions.json` | Toolchain/runtime installation and pinned versions. |
| `download-checkpoint.py`, `check-kernels.*`, `check-lan-exposure.py`, `check-secret-hygiene.py`, `capture-runtime.py` | Setup and audit helpers. |
| `edge/` | Reverse-proxy (Caddy) site template and staging/rollback helpers for the edge host. |

The release itself (`harness/release.py`, `harness/release-management.ps1`) stages these files
with per-file hashes and a rollback manifest; see `harness/README.md`.

## Placeholders and `local/`

The tracked files are **sanitized copies**: site-specific values were replaced by placeholders.
The owner's working copies with the real values live under `local/` (gitignored) at the same
relative path, e.g. `local/serving/supervisor.ps1`. When deploying from this repository, use the
`local/` copy if it exists, otherwise fill in the placeholders. The values themselves are listed
in `config/local.env` (gitignored; template `config/local.env.example`).

| Placeholder in tracked files | Meaning (`config/local.env` key) |
|---|---|
| `/home/USER/qwen36-api` | Linux runtime root (`QWEN36_ROOT`); `/home/USER` = the WSL user's home |
| `USER` | WSL / Windows user name (`QWEN36_WSL_USER`) |
| `C:\path\to\LLM\qwen36-api` (`/path/to/LLM`) | Windows install root (`QWEN36_WIN_ROOT`) |
| `/path/to/qwen36-workspace`, `C:\path\to\qwen36-workspace` | the previous workspace (`QWEN36_ARCHIVE`); after repointing: this repo (`QWEN36_WORKSPACE`) |
| `llm.example.com`, `forge.example.com`, `example.com` | public HTTPS names (`QWEN36_PUBLIC_HOST`, `QWEN36_FORGE_HOST`) |
| `EDGE_SSH_HOST` | SSH alias of the reverse-proxy host (`QWEN36_EDGE_SSH_HOST`) |
| `192.0.2.x`, `203.0.113.x` | documentation addresses replacing LAN/public IPs |

Secrets (the API key file `api.env`, TLS/DNS credentials) are never part of this repository;
the unit reads them from `${QWEN36_SECRETS_DIR}` at run time.

Replacing the placeholders by reads of `config/local.env` / environment variables is an open
TODO (see `MANIFEST.md`).
