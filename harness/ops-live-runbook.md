# Живая эксплуатационная приёмка qwen36: runbook для родителя после выпуска

Статус документа: **подготовлено, ничего из live-шагов не выполнялось** (2026-10-13). Выполнены только CPU/mock-тесты и синтетические файловые пробы; результаты и находки — в [ops-live-audit.json](ops-live-audit.json). Внешний end-to-end — **НЕ ПРОВЕРЕНО** (нет внешней машины), инструкция для владельца в разделе 9.

Ревизия 2 (2026-10-13, после независимой проверки): R-01 — `ops-live-probe.py overload` больше не может принять естественное завершение своего запроса за отмену (D4: `max_tokens=8192` + `ignore_eos`, порог освобождения 5 с как критерий FAIL, проверка `release_attributable_to_cancel`); R-02 — добавлена фаза C2: освобождение GPU и выход keepalive **сразу после смерти supervisor**, до любого нового старта (новый `ops-live-owner-fault.ps1`). Harness фазы C этого не проверяет.

Ревизия 3 (2026-10-14, ops-fixes; только staged-исходники и CPU-тесты, live не выполнялся): F-01..F-03 исправлены (retry `File.Replace` + терпимость heartbeat ≤20 с, shared-read в `manage.ps1`/`test-ops-v2.ps1`, удаление state у keepalive только после двух промахов ≥0,2 с); F-04, F-05, F-06, F-07, F-11 исправлены в harness/readiness; network-acceptance проверяет 400 guard v2 и привязан к instance (F-17); snapshot сверяет все 11 managed-файлов и дерево `release/` с `release/qualification.json` (RP-1). Исправления действуют на live только **после установки нового release** (контроллер `C:\path\to\LLM\qwen36-api` и native обновляются release-инсталляторами; `qualification.json` нужно создавать уже по новым байтам `keepalive.py`/`readiness.py`). F-08, F-09 — по-прежнему смягчения runbook (D4, C2).

Область: ручной сервис `qwen36-inference.service` с моделью `qwen36-35b` (checkpoint `nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a…`, все 256 экспертов, top‑k 8), Windows supervisor/watcher/keepalive, reverse SSH tunnel на `EDGE_SSH_HOST` (192.0.2.10), Caddy `https://llm.example.com/v1`, прежний сайт `forge.example.com`.

## 0. Жёсткие правила на всё окно приёмки

- Только **после** выпуска: Linux `release.py install` и Windows `release-management.ps1` уже выполнены, оба rollback manifest сохранены (пути из их вывода), `release/qualification.json` существует, проверяемый профиль `qualified=true`. Без этого шаги ниже не начинать: на сегодня установлен старый ручной 64K (контроллер без watcher/host-metrics, в native нет `keepalive.py`, `ingress.py`, `profile-runtime.py`, `release/`).
- Эксклюзивное окно: никаких GPU-экспериментов других агентов, `qwen36-experiment.service` неактивен, пользовательской нагрузки нет. Во время подготовки этого runbook другой агент запускал `qwen36-experiment.service` — это нормально для разработки, но **несовместимо** с живой приёмкой.
- Запрещено на любом шаге: `wsl --shutdown/--terminate`, перезагрузки, `taskkill`/`pkill`/`killall` по имени, ручное удаление `C:\path\to\LLM\qwen36-api\run\supervisor.json`, бесконечные `reset-failed`, правка Caddy/DNS/firewall, печать ключа. Ключ не вставлять в команды, отчёты, чат.
- Сбой любого шага = FAIL этого шага с сохранённым receipt. Не ослаблять пороги, не «перезапускать до зелёного». Допустим **один** повтор только для транспортного сбоя, оба receipt сохраняются.
- Во время `test-ops-v2.ps1 -Live` и worker-fault не запускать параллельно `qwen36 status/start/stop`: читать состояние можно только `ops-live-snapshot.ps1` (shared read). После ревизии 3 Windows-читатели не ломают heartbeat (F-01/F-02 исправлены), но правило сохраняется ради однозначных receipts и control-мьютекса.

### 0.1 Известные дефекты, которые нужно было решить ДО живой приёмки

Статус 2026-10-14: **исправлено в staged-исходниках** (ops-fixes), CPU-тесты PASS, live NOT_RUN. Пока установлен старый контроллер, таблица ниже описывает установленное поведение.

| ID | Где | Что будет на live | Решение |
|---|---|---|---|
| F-01 (high) | `supervisor.ps1:170` (Save-State → `File.Replace` без retry; вызов в цикле :239; в ревизии 3 — `Save-State` с retry 20×50 мс, цикл вызывает `Save-Heartbeat`) | Любой Windows-читатель `supervisor.json` без `FILE_SHARE_DELETE` (`manage.ps1:9`, `test-ops-v2.ps1:38`) может вызвать `ERROR_SHARING_VIOLATION` → исключение в главном цикле → supervisor **останавливает весь сервис**. Воспроизведено синтетически: 8/537 и 19/1063 Replace упали; с shared-reader 0/800 | **Сделано:** retry 20×50 мс на `IOException`/`UnauthorizedAccessException`; только периодический heartbeat терпит сбой, пока последняя успешная запись моложе 20 с (stale keepalive 30 с), иначе fail-closed; записи ownership/intent/child остаются строгими. Стресс (свой writer-процесс, 5 с): 0 сбоев Replace при читателе без `FILE_SHARE_DELETE` |
| F-02 (medium) | `manage.ps1:9`, `test-ops-v2.ps1:38` | Сам `Get-Content` получает IOException во время Replace (16/1899, 26/3790 в стресс-пробе) → ложный FAIL harness/`qwen36 status` | **Сделано:** `Read-SharedText` (`FileShare ReadWrite\|Delete`, 5×50 мс) в `manage.ps1` (Read-State и вывод `status`) и `test-ops-v2.ps1` |
| F-03 (medium) | `keepalive.py:109-110` | DrvFS кратко отдаёт ENOENT во время Replace (2–8 ENOENT за пробу, 0 после повторной проверки через 0,2 с) → helper выходит с `owner_state_removed` при живом owner, supervisor пересоздаёт Keepalive, расходуя лимит 6/15 мин | **Сделано:** `owner_state_removed` только после второго промаха через ≥0,2 с (`MISSING_CONFIRM_SECONDS`); промах не продлевает lease (stale 30 с) |

Если приёмка всё же идёт на контроллере без ревизии 3: любой FAIL, у которого в snapshot `SupervisorLog.ReplaceFailureSeen=true` или в keepalive receipts `exit_reason=owner_state_removed` при ещё существующем owner, классифицировать как проявление F-01/F-03 (известный дефект), а **не** как PASS и не как «случайность».

## 1. Переменные и вспомогательные команды

PowerShell 5.1 из корня workspace:

```powershell
$WS   = 'C:\path\to\qwen36-workspace'
$OPT  = "$WS\outputs\qwen36-api\optimization"
$RCPT = "$OPT\ops-live-receipts"            # каталог snapshot по умолчанию
$PY   = '/home/USER/qwen36-api/.venv/bin/python'
$LOPT = '/path/to/qwen36-workspace/outputs/qwen36-api/optimization'
$PROFILE_NAME = 'main128'                    # проверяемый qualified профиль
function Snap([string]$Label,[string[]]$Extra=@()) {
  $out = @(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$OPT\ops-live-snapshot.ps1" -Label $Label @Extra)
  $code = $LASTEXITCODE; $out | Out-Host
  if ($code -ne 0) { Write-Warning "snapshot exit $code (2 = неполный снимок, 1 = вердикт FAIL)" }
  (($out -join "`n") | ConvertFrom-Json).Snapshot }
function Probe { wsl.exe -d archlinux -u USER -- env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy $PY "$LOPT/ops-live-probe.py" @args }
function Inspect([string]$G,[string]$L,[string]$Expect,[int]$Wait=0) {
  wsl.exe -d archlinux -u USER -- $PY "$LOPT/inspect-keepalive.py" --generation $G --lease-id $L --expect $Expect --wait-seconds $Wait }
```

Новые инструменты этой задачи (только чтение/синтетика, кроме явно live-режимов):

- `ops-live-snapshot.ps1` — read-only снимок: owner identity и живость PID+StartTime, обе units, nvidia-smi, хэши установленных Windows controller и всех 11 native managed-файлов против staged, `ReleaseTree` (каждый `source_files` из установленного `release/qualification.json` совпадает по sha256, в `release/` нет лишних и нерегулярных записей кроме `__pycache__`/`*.pyc`, установленный receipt = staged `outputs/qwen36-api/qualification.json`), классификация `supervisor.log` без сырого текста; `-CompareStoppedBaseline` даёт вердикт освобождения GPU. Читает state через `FileShare ReadWrite|Delete` (не провоцирует F-01).
- `ops-live-probe.py` — по умолчанию PLAN_ONLY. `identity --live` (loopback `/health` + `/v1/stats`: модель, непустой и совпадающий instance, `stats.model.ctx` = контекст профиля, файл `selected-profile`, units), `overload --live` (через `EDGE_SSH_HOST`: 429 ingress burst, 413 >2 MiB на tunnel и HTTPS, 429 busy nonstream/stream c `Retry-After` до SSE, освобождение слота ≤5 с после закрытия своего SSE и nonstream соединения и раньше, чем свой запрос мог бы завершиться сам; identity до/после), `keepalive-gone --live` (read-only, все receipts поколений, сигналов нет).
- `ops-live-owner-fault.ps1` — без `-Live` только чтение (PLAN_ONLY: проверка предусловий и точной цели). С `-Live` отправляет ровно один сигнал: TerminateProcess проверенного по PID+StartTime (handle удерживается) `Supervisor` или `Watchdog` ожидаемого поколения, под тем же `Global\Qwen36ApiControl`, что `qwen36 start/stop`; owner-мьютекс `Global\Qwen36ApiSupervisor` не берёт (его берёт watcher для уборки). Дальше только наблюдает уборку: state удалён, все 5 identity исчезли, unit `inactive/failed` с `MainPID=0`, без воскрешения 15 с. Units не запускает и не останавливает, state не пишет и не удаляет, секреты не читает. GPU не оценивает — это делает следующий snapshot compare.
- CPU-тесты: `test-ops-live-probe-cpu.py` (37), `test-ops-live-snapshot-cpu.ps1` (38), `test-ops-live-owner-fault-cpu.ps1` (23, только свои синтетические процессы и приватный Local-мьютекс).

Отчёты probe: `/home/USER/qwen36-api/optimization/reports/ops-live-<mode>-<UTC>-<id>.json` (0600, O_EXCL). Snapshot: `$RCPT\ops-live-snapshot-<label>-<UTC>-<id>.json` (CreateNew). Owner-fault: `$RCPT\ops-live-owner-fault-<label>-<UTC>-<id>.json` (CreateNew; exit 0 = PASS/PLAN_ONLY, 1 = FAIL, 2 = REFUSED — сигнал не отправлялся).

```powershell
function OwnerFault([string]$Target,[string]$Generation,[string]$Label,[switch]$Live) {
  $a = @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$OPT\ops-live-owner-fault.ps1",'-Target',$Target,'-ExpectGeneration',$Generation,'-Label',$Label)
  if ($Live) { $a += '-Live' }
  $out = @(powershell.exe @a); $code = $LASTEXITCODE; $out | Out-Host
  if ($code -ne 0) { Write-Warning "owner-fault exit $code (1 = FAIL, 2 = REFUSED)" }
  ($out -join "`n") | ConvertFrom-Json }
```

## 2. Фаза A — предварительные проверки без сервиса (≈10 мин)

**A1. GPU-окно.** В отдельном терминале WSL удерживать lock на всё окно (свой процесс, ограничен 4 ч):
```bash
flock -n /home/USER/qwen36-api/run/gpu-exclusive.lock -c 'sleep 14400'
```
Ожидание: команда висит. FAIL: немедленный выход (lock занят другим агентом) → ждать/согласовать, не начинать. Освобождение: Ctrl+C в этом терминале (завершает только свой `sleep`).

**A2. CPU-регресс** (≈2 мин): как в разделе 10. Ожидание: числа PASS как в таблице раздела 10. FAIL: любой упавший тест → стоп.

**A3. Snapshot до старта:**
```powershell
$preStart = Snap 'pre-start'
```
Ожидание (exit 0, `Errors` пуст): `Owner.Present=false`; inference `inactive/dead`, `MainPID=0`; experiment `inactive` или `not-found`; все 5 `ControlHashes.*.Match=true`; `NativeHashes.*.Match=true` для всех 11 файлов и `ReleaseTree.Match=true` (`Reason=null`, `Mismatched`/`Missing`/`Extra` пусты, `NonRegular=0`, `ManagedSetMatches=true`, `QualificationStagedMatch=true`) — иначе установлен не тот release или правились staged-файлы после выпуска; `Gpu[0].UsedMiB` — базовая линия (сегодня без модели наблюдалось 3255 MiB и 15049 MiB при чужом эксперименте; база обязана сниматься при неактивном эксперименте). FAIL: любое несоответствие → стоп, разбор release receipts. Receipt: файл snapshot.

## 3. Фаза B — ручной lifecycle через `qwen36` (≈20–25 мин)

| Шаг | Команда | Ожидание | FAIL | Безопасная остановка | Receipt | Время |
|---|---|---|---|---|---|---|
| B1 start | `qwen36 start --profile $PROFILE_NAME` | exit 0, «Supervisor ownership established…» ≤15 с. Это не readiness | ненулевой exit, «did not publish», «experiment is active», «Profile selection failed» | `qwen36 stop` | вывод команды, snapshot B3 | <1 мин |
| B2 readiness | `$t0=Get-Date; do { $r = wsl.exe -d archlinux -u USER -- $PY /home/USER/qwen36-api/readiness.py --edge; $c=$LASTEXITCODE; if($c){Start-Sleep 10} } until($c -eq 0 -or ((Get-Date)-$t0).TotalSeconds -gt 600); $r; qwen36 status` (опрос readiness напрямую не читает `supervisor.json`, F-01/F-02; один финальный `qwen36 status` — receipt) | последний JSON и `qwen36 status`: `ready=true`, `backend_ready=true`, `route.tunnel.ready=true`, `route.https.ready=true`, `model=qwen36-35b`, `profile=$PROFILE_NAME`, `instance_id` непустой и равен `route.tunnel.instance_id`, `profile_verified=true` (`served_context` = контекст профиля), exit 0 | 600 с без ready; `generation_mismatch`/`generation_unverified`; `profile_verified=false`; `error_type` | `qwen36 stop` | последний JSON status | 2–6 мин |
| B3 identity | `Probe identity --live --profile $PROFILE_NAME` | `pass=true`; `served_context_matches_profile` (131072/262144/65536); `instance_id` = из B2; `requests_active=0` | любой row `pass=false` (особенно ctx или пустой instance — закрывает F-05/F-06) | `qwen36 stop` | `ops-live-identity-*.json` | <10 с |
| B4 snapshot running | `$run1 = Snap 'running-1'` | `Owner.Present`, `LifetimeJobArmed=true`, `ModelStartRequested=true`, Phase `model-loading-or-serving`, все 5 процессов `Alive=true`, inference `active` | процесс не жив, флаги false | `qwen36 stop` | snapshot | <30 с |
| B5 keepalive running | `$s=ConvertFrom-Json (Get-Content $run1 -Raw); $g1=$s.Owner.Generation; $l1=$s.Owner.KeepaliveLeaseId; Inspect $g1 $l1 running` | `pass=true`, `exact_identity_alive=true`, `signals_sent=0` | missing/ambiguous receipt, не жив | `qwen36 stop` | JSON вывода (сохранить в файл) | <10 с |
| B6 churn 5 мин | подождать 5 мин без чтений state; `$run1b = Snap 'running-1b'` | тот же Generation и **тот же** `KeepaliveLeaseId`, те же Keepalive/Tunnel PID+Started, `SupervisorLog.ReplaceFailureSeen=false`, `RestartLimitSeen=false` | lease сменился (F-03), Replace failure (F-01), supervisor исчез | `qwen36 stop` | snapshot | 5 мин |
| B7 повторный start | `qwen36` (без аргументов) | exit 0, «Supervisor is running; inference start requested»; затем `Probe identity --live --profile $PROFILE_NAME --expect-instance <instance B2>` PASS | новый instance/generation | `qwen36 stop` | вывод + identity | <1 мин |
| B8 restart | `qwen36 restart`, затем повторить B2, `Probe identity --live --profile $PROFILE_NAME --different-from <instance B2>` | новый Generation в snapshot, новый instance, ready; `Probe keepalive-gone --live --generation $g1` PASS (helper старого поколения исчез) | старый owner/helper жив, instance прежний | `qwen36 stop` | status JSON, identity, keepalive-gone | 3–6 мин |
| B9 stop | `$run2 = Snap 'running-2'` (перед stop), `qwen36 stop`; затем `qwen36 status` | stop: exit 0 «Owned service and supervisor stopped»; status: exit ≠0, `Supervisor running: False` | stop >75 с, «stop failed; state retained» | повторить `qwen36 stop` один раз; если unit остаётся active: `wsl.exe -d archlinux -- systemctl --user stop qwen36-inference.service` (только named unit) | вывод | ≤1,5 мин |
| B10 GPU/keepalive после stop | `$st1 = Snap 'stopped-1' @('-CompareStoppedBaseline',$preStart,'-RunningSnapshot',$run2)`; `Probe keepalive-gone --live --generation <Generation running-2>` | snapshot exit 0, `StoppedReleaseVerdict.Pass=true` (owner отсутствует, unit inactive и `MainPID=0`, experiment неактивен, GPU ≤ база+1024 MiB, освобождено ≥6000 MiB от running); keepalive-gone PASS | Pass=false; helper жив | см. раздел 8 | snapshot, отчёт probe | ≤2 мин |

Если `GpuWithinBaselineTolerance=false`, но `GpuReleasedFromRunning=true`: через 60 с снять ещё один snapshot и проверить посторонние GPU-приложения Windows (база «без модели» сегодня колебалась 2476→3255 MiB). Это не PASS, пока причина не объяснена; порог не увеличивать задним числом.

Опционально, для каждого дополнительного qualified профиля (`long256`, `baseline64`): `qwen36 start --profile <P>` → B2 → `Probe identity --live --profile <P>` → `qwen36 stop` → B10. long256 грузится дольше и даёт меньший запас VRAM.

## 4. Фаза C — harness: tunnel death, HostMetrics, supervisor death (≈20–30 мин)

Предусловие: B10 PASS, сервис остановлен, `stop.request` отсутствует. Параллельно `qwen36 status` не запускать.

```powershell
powershell.exe -NoProfile -File "$OPT\test-ops-v2.ps1" -Live -IncludeFaults -Profile $PROFILE_NAME -ReadyTimeoutSeconds 600
```

Ожидание: итог `Result=PASS`, отчёт `$OPT\ops-v2-live-<UTC>-<id>.json`. В `Events`: `installed_control_hashes`, `manual_only_known_startup_paths`, `manual_start_ready`, `idempotent_start_ready`, `manual_restart_ready`, `old_generation_stop_ignored`, `old_generation_stop_backend_still_ready`, `crash_Tunnel` INJECTED, `tunnel_outage_readiness`, `recovered_Tunnel`, `tunnel_recovery_ready`, `crash_HostMetrics`, `recovered_HostMetrics`, `hostmetrics_recovery_ready`, `crash_Supervisor`, `owner_death_watcher_cleanup`, `no_resurrection_after_owner_death_15s`, `manual_start_after_owner_death`, `final_stop`, `stopped_status_nonzero`, `lifecycle_acceptance` — все PASS.

Обязательные проверки отчёта (п. 1–2 ревизия 3 делает в harness, но их нужно прочитать в receipt):

1. Верхнеуровневое `TunnelOutageObserved`: `true` — outage реально наблюдался и readiness был not-ready; `false` (событие `NOT_OBSERVED_ALREADY_RECOVERED`) — **восстановление tunnel PASS, обнаружение outage НЕ ПРОВЕРЕНО** (F-07). Нельзя записывать как «tunnel outage detection PASS».
2. Во всех `*_ready` событиях `Data.InstanceId` непустой (harness ревизии 3 сам падает на пустом id и на `profile_verified≠true`, F-05/F-06); `manual_restart_ready` и `manual_start_after_owner_death` имеют новые instance; `tunnel_recovery_ready`/`hostmetrics_recovery_ready` — тот же instance, что `manual_restart_ready`.
3. Keepalive всех поколений harness исчез:
   ```powershell
   $ops = (Get-ChildItem "$OPT\ops-v2-live-*.json" | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
   $opsWsl = '/mnt/' + $ops.Substring(0,1).ToLower() + '/' + $ops.Substring(3).Replace('\','/')
   Probe keepalive-gone --live --from-ops-report $opsWsl --wait-seconds 45
   ```
   Ожидание `pass=true`; для поколения supervisor crash записать `exit_reason` (`owner_state_removed` или `heartbeat_stale` допустимы; `receipt_state=running` при мёртвом PID означает, что helper убит сигналом без записи — PASS по identity, но отметить).
4. GPU после harness: `$st2 = Snap 'after-harness' @('-CompareStoppedBaseline',$preStart,'-RunningSnapshot',$run2)` → `StoppedReleaseVerdict.Pass=true`, `SupervisorLog.ReplaceFailureSeen=false`, `OwnerDeathCleanup≥1`, `StopUnconfirmedSeen=false`. Это проверка **финальной остановки** (`final_stop`), а не смерти владельца: после `crash_Supervisor` harness через 15 с сам грузит модель заново (`test-ops-v2.ps1:333-339`, F-09 не исправлен в harness), snapshot в это окно не попадает. Освобождение GPU и выход keepalive после смерти supervisor доказывает только фаза C2.

FAIL: `Result=FAIL` (смотреть `Events[-1].Data.Line`), любой из пунктов 1–4. Безопасная остановка: harness сам в `finally` останавливает **своё** поколение; при `CleanupRefused`/`OwnedCleanupFailed` выполнить `Snap 'harness-fail'`, затем `qwen36 stop` (Cleanup-Orphan адресно останавливает unit и известные identity). Не удалять state, не убивать по имени. Если harness-окно закрыто принудительно — supervisor остаётся владельцем, остановка `qwen36 stop`. Receipts: ops-v2 отчёт, keepalive-gone, snapshot, `C:\path\to\LLM\qwen36-api\logs\supervisor.log` (локально, без публикации).

## 4a. Фаза C2 — смерть supervisor и освобождение GPU вне harness (≈10–12 мин)

Зачем: HANDOFF-OPS-TO-CLAUDE.md:29 и :132 требуют живого доказательства цепочки «смерть владельца → Job убивает своих children → watcher останавливает named unit → Linux keepalive выходит → GPU освобождается». Harness фазы C после `crash_Supervisor` проверяет только исчезновение state, identity и unit, а через 15 с снова грузит модель. Поэтому GPU здесь снимается **до любого нового старта**.

Предусловие: фаза C PASS, сервис остановлен, `qwen36 status` exit ≠0, параллельно ничего не запускать.

| Шаг | Команда | Ожидание | FAIL | Безопасная остановка | Receipt | Время |
|---|---|---|---|---|---|---|
| C2.1 start | `qwen36 start --profile $PROFILE_NAME` → B2 → `Probe identity --live --profile $PROFILE_NAME` | как B1–B3, instance `I_S` | как B1–B3 | `qwen36 stop` | status JSON, identity | 3–6 мин |
| C2.2 running snapshot | `$runS = Snap 'before-supervisor-kill'; $sS = ConvertFrom-Json (Get-Content $runS -Raw); $gS = $sS.Owner.Generation; $odcS = $sS.SupervisorLog.OwnerDeathCleanup` | как B4: все 5 процессов `Alive=true`, `LifetimeJobArmed=true`, `ModelStartRequested=true`, inference `active`, `ReplaceFailureSeen=false` | процесс не жив, флаги false | `qwen36 stop` | snapshot | <30 с |
| C2.3 plan | `$p = OwnerFault Supervisor $gS 'plan-supervisor'` | exit 0, `Result=PLAN_ONLY`, `SignalsSent=0`; в receipt `TargetIdentity` совпадает с `Owner.Processes.Supervisor` из C2.2 | exit 2 `REFUSED` (причина в `Reason`, сигнал не отправлен) | ничего не делать; разобрать причину; `qwen36 stop` | receipt plan | <10 с |
| C2.4 fault | `$f = OwnerFault Supervisor $gS 'supervisor-death' -Live` | exit 0, `Result=PASS`, `SignalsSent=1`; в receipt `Cleanup.CompleteS` ≤150 с (ожидается ~5–40 с: watcher опрашивает раз в 3 с, `TimeoutStopSec=30`), `StateRemovedS`/`IdentitiesGoneS`/`UnitStoppedS` заданы, `NoResurrection.Observations≥1` (15 с) | exit 1: `cleanup_deadline_reached` (watcher не убрал), `new_owner_appeared_during_cleanup`, `unit_resurrected_after_cleanup`, `owner_state_reappeared`, `target_did_not_exit`; exit 2 = сигнал не отправлен | при `cleanup_deadline_reached`: **сначала** C2.5 (снимок факта), затем `qwen36 stop` (Cleanup-Orphan адресно); при `new_owner_appeared_during_cleanup` не вмешиваться, snapshot, эскалация | `ops-live-owner-fault-supervisor-death-*.json` | ≤3 мин |
| C2.5 GPU сразу после смерти владельца | до любого нового start: `$stS = Snap 'after-supervisor-kill' @('-CompareStoppedBaseline',$preStart,'-RunningSnapshot',$runS)` | exit 0, `StoppedReleaseVerdict.Pass=true` (owner state нет, unit `inactive/failed` и `MainPID=0`, experiment неактивен, GPU ≤ база+1024 MiB, освобождено ≥6000 MiB относительно `before-supervisor-kill`); `SupervisorLog.OwnerDeathCleanup` > `$odcS` (уборку сделал именно watcher); `StopUnconfirmedSeen=false` | `Pass=false`; `OwnerDeathCleanup` не вырос | как B10: повторный snapshot через 60 с допустим только для `GpuWithinBaselineTolerance=false` при `GpuReleasedFromRunning=true`, порог не менять; живой unit → `qwen36 stop` | snapshot | <1 мин |
| C2.6 keepalive | `Probe keepalive-gone --live --generation $gS --wait-seconds 45` | `pass=true`; записать `exit_reason` (`owner_state_removed` или `heartbeat_stale` допустимы; `receipt_state=running` при мёртвом PID — helper убит без записи, PASS по identity, отметить) | helper этого поколения жив | `qwen36 stop`; helper вручную не убивать | `ops-live-keepalive-gone-*.json` | ≤1 мин |
| C2.7 status | `qwen36 status` | exit ≠0, `Supervisor running: False` | exit 0 или supervisor жив | — | вывод | <10 с |

Новый `qwen36 start` (фаза D) — только после C2.5–C2.7. Restart budget фазы E это не расходует: supervisor делает `reset-failed` перед каждым ручным стартом (`supervisor.ps1:256`).

## 5. Фаза D — сеть через Caddy (≈10 мин)

**D1. Старт** (после C2.5–C2.7): `qwen36 start --profile $PROFILE_NAME` → B2 → `Probe identity --live --profile $PROFILE_NAME` (запомнить instance `I3`) → `$run3 = Snap 'running-3'`.

**D2. Network acceptance** (live при вызове, dry-run нет):
```powershell
wsl.exe -d archlinux -u USER -- $PY "$LOPT/network-acceptance.py" --name ("network-release-" + (Get-Date -Format yyyyMMddTHHmmss))
```
Ожидание: exit 0, `pass=true`, 60 rows, `same_instance=true` и `instance_id` = `I3`, `tls_verified=true`, `redirects_followed=false`, `external_e2e=NOT_RUN`. Rows: `instance-before`/`instance-after` (loopback `/health`: 200, `qwen36-35b`, валидный instance id) и `same-instance`; `models` 200 на `http://127.0.0.1:1919` (edge loopback → tunnel) и `https://llm.example.com`; `auth-False/auth-wrong` и `auth-chat-*` — ровно **401** (8 строк; 403/429/502/transport = FAIL); `nonstream` и `stream` — ответ `42`, alias `qwen36-35b`, SSE `[DONE]`; 32 строки `unsupported-*` — 400 JSON (`error` объект) без SSE: прежние 12 (`forced-required`, `forced-named`, `json-mode`) и 20 guard v2 (`parallel-false` = `parallel_tool_calls:false` при tools, `parallel-nonbool` = `"false"` строкой, `strict-function`, `strict-tool`, `legacy-functions` без tools) × stream/nonstream × loopback/HTTPS; запросы с паузой 0,12 с, чтобы ingress-429 не подменял 400; `auto-tool-cycle` — один вызов `fixture_add`, валидный id/JSON, финал `42`; `closed-route` 404 ×9 (`/health`, `/v1/stats`, `/v1/cache/status`, `/metrics`, `/docs`, `/v1/responses`, `/v1/completions`, HEAD/OPTIONS); `existing-forge-site` 200. FAIL: любой row `pass=false` или `scope=SSH_OR_REMOTE_CHECK_FAILURE`. Receipt: `/home/USER/qwen36-api/optimization/reports/network-release-*.json`. ≈2–4 мин.

**D3.** `Probe identity --live --profile $PROFILE_NAME --expect-instance <I3>` → PASS (network-acceptance ревизии 3 привязан к instance, но не к профилю/ctx и не проверяет 413/429/отмену: F-08 → D4).

**D4. Перегрузка/413/отмена** — подождать 60 с (восстановление token bucket), затем:
```powershell
Probe overload --live --profile $PROFILE_NAME
```
Ожидание `pass=true`:
- `unauthenticated-burst-ingress-429`: только 401/429, ≥1 ingress 429 «Request rate exceeded».
- `body-over-2MiB-413` на tunnel и HTTPS: status 413, `send_error_type` допустим.
- `own-sse-stream-started`: свой запрос с `own_max_tokens=8192` (максимум guard) и `ignore_eos=true`. Сам он не может завершиться раньше `natural_completion_floor_s` ≈27,3 с даже при 300 ток/с; при измеренных 72–131 ток/с это ≈62–113 с.
- `busy-429-nonstream` и `busy-429-stream-before-sse`: 429, `Retry-After`, JSON, без SSE, без ожидания в очереди.
- `sse-client-close-releases-slot` и `nonstream-client-close-releases-slot` — **критерий FAIL, не ориентир**. Нужны `release_s ≤ release_deadline_s` (5 с от закрытия своего соединения, время берётся по ответу подтверждающего запроса) и `release_attributable_to_cancel=true` (`own_request_age_s < natural_completion_floor_s`). Прежние локальные замеры отмены ≈0,8–1,0 с (CLAUDE-HANDOFF §8) — справка; порог 5 с задним числом не увеличивать. Если SSE-освобождение FAIL, nonstream-кейс не запускается (`skipped=previous_release_failed_slot_may_be_held`, FAIL): вторая длинная генерация поверх возможно застрявшей не ставится. `rate_limited_polls>0` означает, что опрос упирался в ingress bucket; это возможный ложный FAIL, но не PASS.
- `identity_after` тот же instance, `slot_idle_after=true`.

FAIL: любой row; `SERVICE_NOT_IDLE` (повторить после проверки, что нет чужих запросов). Повтор только при `transport_or_schema_failure`, один раз. После FAIL освобождения свой запрос может держать слот до ≈2 мин (8192 токена): фазу E не начинать. `Probe identity --live --profile $PROFILE_NAME` раз в 15 с до `requests_active=0` (не дольше 180 с), иначе `qwen36 restart` → B2. 408 slow-body через Caddy этим шагом **не проверяется** (NOT_RUN; локально ранее PASS в isolated стенде). ≈2–3 мин.

## 6. Фаза E — worker faults (≈25–35 мин)

Предусловия: сервис из D готов, `identity` PASS, `requests_active=0`, нет внешних клиентов и параллельных ops-команд. Порядок строго последовательный. Restart budget unit: 3 старта за 900 с (`qwen36-inference.service:4-5`); manual start + SIGKILL + SIGSTOP = ровно 3 → между ними обязателен `qwen36 restart` (сбрасывает счётчик через `reset-failed`), иначе лишний crash-restart сделает FAIL stall-теста.

**E1. SIGKILL idle worker:**
```powershell
wsl.exe -d archlinux -u USER -- env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy $PY "$LOPT/test-worker-fault-v2.py" --live --case kill
```
(`env -u` оставлен как дополнительная защита: в ревизии 3 скрипт уже использует прямой opener без proxy и redirect, F-04.) Ожидание: `result=PASS`, `after.instance_id` ≠ before, `after.invocation_id` ≠ before, `after.nrestarts` > before, `generation_http=200`, `cleanup.errors=[]`. FAIL: `model_worker_is_not_unique`, `automatic_recovery_deadline` (480 с) и др. Остановка: сервис оставлен как есть → `qwen36 status`; при неготовности `qwen36 stop`. Receipt: `/home/USER/qwen36-api/reports/worker-fault-kill-*.json`. ≈3–9 мин.

**E2.** `qwen36 status` → ready через Caddy (новый instance виден в `route.tunnel.instance_id`); `Probe identity --live --profile $PROFILE_NAME --different-from <I3>` PASS.

**E3.** `qwen36 restart` → B2 → identity (instance `I5`).

**E4. SIGSTOP > 300 с (stall detector):**
```powershell
wsl.exe -d archlinux -u USER -- env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy $PY "$LOPT/test-worker-fault-v2.py" --live --case stall
```
Ожидание: `result=PASS`, `stopped_until_worker_exit_seconds` в [300, 370], `stall_shutdown_event=true`, новый instance/invocation, `cleanup.errors=[]`. FAIL без переинтерпретации: `worker_exited_before_expected_300_second_stall`, `stall_detector_did_not_remove_worker_before_resume_deadline`, `missing_current_300s_stall_shutdown_event`. Прерывание: только Ctrl+C этого процесса (finally и независимый guard отправят SIGCONT через pidfd не позже 380 с). Receipt: `worker-fault-stall-*.json`. ≈8–15 мин.

**E5.** `qwen36 status` ready; `Probe identity --live --profile $PROFILE_NAME --different-from <I5>` PASS.

**E6 (опционально) — смерть watcher (fail-closed).** Ожидаемое поведение: supervisor за ≤5 с замечает исчезновение watcher, штатно останавливает unit, удаляет state и выходит. Только адресно, тем же инструментом (PID+StartTime с удерживаемым handle, под control-мьютексом):
```powershell
$runW = Snap 'before-watcher-kill'; $gW = (ConvertFrom-Json (Get-Content $runW -Raw)).Owner.Generation
$p = OwnerFault Watchdog $gW 'plan-watchdog'              # PLAN_ONLY, SignalsSent=0
$f = OwnerFault Watchdog $gW 'watcher-death' -Live        # PASS: уборка ≤150 с, без воскрешения 15 с
```
Сразу, до любого нового start: `Snap 'after-watcher-kill' @('-CompareStoppedBaseline',$preStart,'-RunningSnapshot',$runW)` → `StoppedReleaseVerdict.Pass=true`, `SupervisorLog.WatcherExitSeen=true`; `Probe keepalive-gone --live --generation $gW --wait-seconds 45` PASS; `qwen36 status` exit ≠0. FAIL: owner-fault exit 1 (сервис продолжает работать без watcher или уборка не завершена), verdict false. Безопасная остановка: snapshot → `qwen36 stop`. Для продолжения нужен новый ручной `qwen36 start`. Receipts: оба owner-fault receipt, snapshots, keepalive-gone. ≈8 мин.

## 7. Фаза F — финальная остановка и GPU (≈3 мин)

`$run4 = Snap 'running-final'` (если сервис запущен) → `qwen36 stop` → `qwen36 status` (exit ≠0) → `$final = Snap 'final-stopped' @('-CompareStoppedBaseline',$preStart,'-RunningSnapshot',$run4)` → `Pass=true`. Затем `Probe keepalive-gone --live --generation <каждое поколение фаз D–E из snapshot>` → PASS. Освободить A1 lock (Ctrl+C). Модель оставить выключенной, если владелец не просил иного.

## 8. Сбой посередине: что останется и как убрать

| Ситуация | Что может остаться | Действие |
|---|---|---|
| Ошибка до публикации owner | ничего (supervisor exit 1, state не создан или удалён) | `qwen36 status`, при остатке `qwen36 stop` |
| owner-fault exit 2 `REFUSED` | ничего не изменено, сигнал не отправлен | разобрать `Reason`/`Preconditions`; проверку не обходить, вручную по PID не убивать |
| owner-fault `FAIL cleanup_deadline_reached` | supervisor (или watcher) мёртв, уборка не завершена: state и/или unit остались, GPU может быть занята | snapshot (receipt факта) → `qwen36 stop` (Cleanup-Orphan) → snapshot compare; шаг остаётся FAIL |
| owner-fault `FAIL new_owner_appeared_during_cleanup` | появился другой владелец (чужой запуск) | не вмешиваться в чужое поколение; snapshot; эскалация |
| overload: освобождение слота FAIL | свой длинный запрос (8192 токена) может держать слот до ≈2 мин | ждать `requests_active=0` (identity, ≤180 с), иначе `qwen36 restart`; фазу E не начинать |
| Supervisor упал (в т.ч. F-01) | watcher останавливает named unit, удаляет state; Windows children убивает Job; Linux keepalive выходит ≤30 с | snapshot + `keepalive-gone`; новый `qwen36 start` только после разбора |
| Watcher не смог остановить unit (3 попытки) | state сохранён, unit может держать GPU | `qwen36 stop` (Cleanup-Orphan: адресная уборка + `systemctl --user stop`) |
| Harness убит/окно закрыто | живой supervisor своего поколения, возможно `stop.request` с `AcceptanceTestId` (игнорируется новыми owner) | `qwen36 stop`; токен уберёт следующий stop/start |
| Worker test FAIL | сервис в любом состоянии, restart budget частично израсходован | `qwen36 status`; `qwen36 stop`; не делать циклы `reset-failed` |
| SIGSTOP tester убит вместе с guard / весь WSL упал | остановленный worker держит GPU до SIGKILL systemd | `qwen36 stop` (TimeoutStopSec=30 → SIGKILL cgroup), затем snapshot compare |
| `StartLimitBurst` исчерпан | unit `failed`, GPU свободна | сохранить receipt, `qwen36 stop`, разбор причины; не обходить |
| Tunnel на edge не освобождён после kill ssh | новые ssh падают ExitOnForwardFailure; после 6 попыток за 15 мин supervisor остановит сервис (fail-closed) | snapshot `RestartLimitSeen`; `qwen36 stop`; повтор позже |
| WSL broker завис | синхронный `wsl.exe` в controller не имеет timeout (F-13) | ждать; не `wsl --terminate`; эскалировать владельцу |

Rollback релиза — только по точным manifest из receipts выпуска: `release-management.ps1 -RollbackManifest <manifest> -CheckOnly`, затем без `-CheckOnly`; `release.py rollback --manifest <manifest> --check`, затем без `--check` при остановленных units ([OPERATIONS.md](OPERATIONS.md)). Live-тесты сами ничего не устанавливают.

## 9. Внешний E2E (владелец, мобильный интернет) — сейчас НЕ ПРОВЕРЕНО

Подготовка (сервер запущен и `qwen36 status` ready): ключ получить локально `qwen36 key` (clipboard), перенести на Arch вручную безопасным способом (менеджер паролей/ручной ввод), **не** через чат, почту, сторонние сервисы. Скопировать на Arch `outputs/qwen36-api/test-external.sh` (ключа в файле нет).

На Arch:
1. Отключить Wi‑Fi домашней сети, включить точку доступа телефона. Проверить, что маршрут не домашний: `ip route` (default не через 192.168.0.x) и `getent hosts llm.example.com` (публичный адрес). Если адрес/маршрут домашние — тест не внешний.
2. `bash test-external.sh`, ввести ключ в скрытый prompt.
3. Ожидание: (а) JSON `/v1/models` содержит `"id":"qwen36-35b"`; (б) JSON ответа с `"model":"qwen36-35b"` и русским текстом; (в) поток строк `data: {...}`, завершающийся `data: [DONE]`. Скрипт стоп при HTTP ≥400 (`--fail-with-body`, `set -e`).
4. Дополнительно (без ключа): `curl -sS -o /dev/null -w '%{http_code}\n' https://llm.example.com/v1/models` → `401`; `curl -sS -o /dev/null -w '%{http_code}\n' https://llm.example.com/health` → `404`; `curl -sS -o /dev/null -w '%{http_code}\n' https://forge.example.com/` → `200`. TLS проверяется curl по умолчанию — не добавлять `-k`.
5. Записать: дата/время, тип сети, коды/признаки (а)–(в) и п.4; без ключа и текстов ответов. Только после этого `external_e2e=PASS`. Скрипт сам не сверяет id/`[DONE]` автоматически (F-14) — проверка глазами обязательна.

## 10. CPU-регресс (воспроизводит то, что выполнено при подготовке)

```powershell
# Исторические receipts ops-control-tests.json и ops-readiness-tests.json эти тесты перезаписывают (F-19):
# запускать изолированно как в work/agents/ops-live (run-cpu-tests.ps1) или принять перезапись осознанно.
powershell.exe -NoProfile -File $OPT\test-ops-v2-cpu.ps1
python -B $OPT\test-worker-fault-v2-cpu.py
python -B $OPT\test-network-acceptance-cpu.py
python -B $OPT\test-readiness-transport-cpu.py
python -B $OPT\test-keepalive-cpu.py
python -B $OPT\test-ops-live-probe-cpu.py
powershell.exe -NoProfile -File $OPT\test-ops-live-snapshot-cpu.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $OPT\test-ops-live-owner-fault-cpu.ps1
```

| Набор | Заявлено в HANDOFF | 2026-10-13 Windows | 2026-10-13 WSL venv | 2026-10-14 ревизия 3 |
|---|---|---|---|---|
| test-ops-v2-cpu.ps1 | 18 | 18 PASS | — (PowerShell) | 30 PASS |
| ops-control-tests.ps1 (изолированная копия, отличия только scratch/receipt) | 39 | 39 PASS | — | 48 PASS |
| test-worker-fault-v2-cpu.py | 16 | 16 PASS | 16 PASS | 19 PASS (WSL) |
| test-network-acceptance-cpu.py | 14 | 14 PASS | 14 PASS | 19 PASS (WSL) |
| ops-readiness-tests.py (запись receipt перенаправлена) | 11 | 11 PASS | 11 PASS | 24 PASS (WSL) |
| test-readiness-transport-cpu.py | 7 | 7 PASS | 7 PASS | 8 PASS (WSL) |
| test-keepalive-cpu.py | 15 | 15 PASS | 15 PASS | 19 PASS (WSL) |
| test-ops-v2.ps1 без -Live (parser/plan) | 5 | STATIC_PASS 5 | — | STATIC_PASS 5 |
| test-ops-live-probe-cpu.py (новый; ревизия 2: +6 тестов R-01) | — | 37 PASS | 37 PASS | 37 PASS (WSL) |
| test-ops-live-snapshot-cpu.ps1 (новый) | — | 24 PASS | — | 38 PASS |
| test-ops-live-owner-fault-cpu.ps1 (новый, ревизия 2, R-02) | — | 23 PASS (4 прогона подряд) | — (PowerShell) | 23 PASS |
| test-lifetime-job.ps1 -RunOwnChildren (свои синтетические процессы) | 7 | — | — | 7 PASS |
| test-release-management-cpu.ps1 | 27 | — | — | 27 PASS |

Ревизия 3 запускалась изолированно: `work/agents/ops-fixes/run-cpu-windows.ps1` и `wsl-cpu-tests.sh` (receipts ops-control/ops-readiness/lifetime-job не перезаписаны).

## 11. Итоговая таблица receipts для выпуска

| Пункт приёмки | Receipt | Статус сейчас |
|---|---|---|
| Установленный release = receipt | snapshot `pre-start`: `NativeHashes` 11/11, `ReleaseTree.Match=true` | NOT_RUN |
| Ручной start/status/restart/stop | snapshots `pre-start`, `running-1(b)`, `running-2`, `stopped-1`; identity; status JSON | NOT_RUN |
| Readiness backend/tunnel/HTTPS | `qwen36 status` JSON, identity | NOT_RUN |
| Tunnel/HostMetrics/supervisor death (harness: уборка, без воскрешения, ручной перезапуск) | `ops-v2-live-*.json` (+ отметка NOT_OBSERVED при быстром восстановлении) | NOT_RUN |
| Supervisor death → GPU освобождена, keepalive вышел (до нового старта) | `ops-live-owner-fault-supervisor-death-*.json`, snapshots `before-supervisor-kill`/`after-supervisor-kill` (verdict), keepalive-gone поколения `$gS` | NOT_RUN |
| Keepalive exit | inspector JSON, `ops-live-keepalive-gone-*.json` | NOT_RUN |
| GPU release | `StoppedReleaseVerdict` в snapshots `stopped-1` (штатный stop), `after-supervisor-kill` (смерть владельца), `after-harness` (финальный stop harness), `after-watcher-kill` (опц.), `final-stopped` | NOT_RUN |
| Caddy 401/400 (forced/json + guard v2)/404/SSE/tool cycle/forge 200, один instance | `network-release-*.json` (60 rows) | NOT_RUN |
| 413/429/отмена/ingress 429 | `ops-live-overload-*.json` (отмена: `release_s≤5`, `release_attributable_to_cancel=true`) | NOT_RUN |
| Worker SIGKILL / SIGSTOP | `worker-fault-kill-*.json`, `worker-fault-stall-*.json` | NOT_RUN |
| Watcher death (опц.) | `ops-live-owner-fault-watcher-death-*.json`, snapshots `before/after-watcher-kill`, keepalive-gone `$gW` | NOT_RUN |
| Внешний E2E | запись владельца по разделу 9 | НЕ ПРОВЕРЕНО |
| Cold boot / автозапуск | не выполняется (автозапуска нет по требованию) | NOT_RUN |

Ориентир общей длительности фаз A–F (включая C2): 1,75–2,25 ч, без внешнего E2E.
