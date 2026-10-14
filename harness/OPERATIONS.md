# Эксплуатация подготовленного релиза Qwen36

Этот документ описывает **staged source**, а не объявляет его установленным. На момент проверки этой инструкции все три записи в `../profiles.json` имеют `qualified=false`; installer и selector обязаны отказать до решения владельца квалификации. Исторические результаты старого 64K сервиса в [README](../README.md) не являются приёмкой новых supervisor, kernels или профилей. Актуальное решение о выпуске и его receipt должны храниться отдельно.

## Ручное управление после квалификации и установки

Контроллер установленного сервиса — `C:\path\to\LLM\qwen36-api\manage.ps1`; короткая команда — `C:\path\to\LLM\qwen36-api\bin\qwen36.cmd`. Команды выполняет владелец Windows/WSL. Секрет не передаётся в argv.

```powershell
qwen36
qwen36 status
qwen36 metrics
qwen36 stop
qwen36 start --profile main128
qwen36 restart --profile baseline64
qwen36 stop
qwen36 start --profile long256
```

Новый обычный `start` выбирает `main128`; bare `qwen36` при уже работающем supervisor сохраняет выбранный профиль и явно запрашивает запуск своей inference unit. `start --profile ...` при живом supervisor отказывает: для смены использовать `restart --profile ...` либо `stop` затем `start`. Непрошедший квалификацию профиль не запускается. Названия означают лимиты prompt+output 131072, 65536 и 262144 соответственно; фактические dtype, flags и библиотеки берутся из установленного `profiles.json`. Документ не выбирает FP8 или CUDA-вариант за владельца квалификации.

`start` возвращается после публикации identity supervisor, **до** завершения загрузки модели. Успешный exit этой команды не является readiness. `status` печатает состояние supervisor и systemd, затем запускает authenticated readiness. При отсутствии живого supervisor exit ненулевой даже если оставшийся backend ещё отвечает. `metrics` читает исторические числовые записи; старый `ready=true` в них не подтверждает текущую готовность.

Readiness на Windows/WSL и на Caddy-host использует прямой urllib opener без inherited proxy и redirects. Bearer не передаётся по `Location` стороннему host; проверка TLS/hostname сохранена. Требуется HTTP 200 с корректной моделью/health, а не просто любой 2xx. Redirect, proxy failure и поколение backend, отличное от tunnel, дают unready.

| Наблюдение | Практический смысл |
|---|---|
| `Phase=starting`, job ещё false | Identity опубликована; watcher/job ещё готовятся |
| `Phase=ownership-ready`, `LifetimeJobArmed=true` | Windows lifetime job установлен, модель ещё может не запускаться |
| `Phase=model-loading-or-serving` | Start запрошен; это поле не различает загрузку и готовность |
| `backend_ready=false`, health `phase/progress` | Модель недоступна или ещё загружается; TCP listener недостаточен |
| `backend_ready=true`, `route.tunnel.ready=false` | Локальная модель готова, путь с Caddy не готов |
| Tunnel ready, HTTPS false | Tunnel проверен, публичное имя/TLS/API с Caddy не прошли |
| Итог `ready=true`, exit 0 | Backend, Caddy tunnel и HTTPS models прошли, совпали доступные instance IDs; это не внешняя генерация |
| Supervisor отсутствует | Ручная сессия закончена либо случился сбой; модель не должна самопроизвольно подняться |

Base URL остаётся `https://llm.example.com/v1`, model id `qwen36-35b`. Доступны только GET `/v1/models` и POST `/v1/chat/completions`. Readiness `/health` используется локально/через Caddy-host loopback, извне этот маршрут закрыт. API-key берётся штатной `qwen36 key` в локальный clipboard; путь секрета — `/home/USER/qwen36-api/secrets/api.env`, 0600. Не вставлять ключ в команды, отчёты или Git.

## Владение процессами и восстановление

`Global\Qwen36ApiControl` сериализует управляющие команды; `Global\Qwen36ApiSupervisor` разрешает единственного owner и блокирует cleanup старого watcher от остановки нового поколения. State в `C:\path\to\LLM\qwen36-api\run\supervisor.json` содержит generation, PID и creation time owner и children. Stop token адресован конкретному owner; чужой/устаревший token игнорируется.

До WSL/model/tunnel supervisor запускает watcher снаружи собственного job и ждёт его handshake до 10 с. Затем он создаёт отдельный unnamed Win32 job с `KILL_ON_JOB_CLOSE`, запрещает наследование единственного job handle и присоединяет себя. Все последующие Windows working children наследуют membership при создании. Поэтому child, созданный прямо перед прерыванием `Save-State`, уже принадлежит job. Проверка membership не подменяет его поздним Assign после CreateProcess.

При смерти owner Windows закрывает его единственный job handle и завершает Windows children/descendants; внешний watcher остаётся для остановки **только** `qwen36-inference.service`. Watcher сверяет PID+creation time и ownership mutex, поэтому старое поколение не может остановить нового owner. `ModelStartRequested` записывается перед WSL start. Если job initialization отказал раньше, watcher не вызывает Linux stop. Несовместимость Win32 job приводит к отказу до модели, а не к запуску без защиты. Подробности и источники Win32: [lifetime-job-review.md](lifetime-job-review.md).

Смерть самого watcher вызывает штатный stop supervisor: replacement изнутри job не пережил бы смерть owner. Следующий ручной start создаёт новую пару owner/watcher. При штатном stop убираются собственные tunnel/metrics, останавливается именованная unit, затем Keepalive/watcher и свой state. При неподтверждённом stop state сохраняется для диагностики/повторной адресной уборки.

| Механизм | Предел |
|---|---|
| systemd inference | `Restart=always`, пауза 20 с, максимум 3 старта за 900 с; явный `systemctl stop` останавливает цикл |
| Tunnel / HostMetrics / Keepalive | Для каждого не более 6 запусков за 15 минут, включая первый; supervisor проверяет раз в 5 с |
| Owner-death watcher | До 3 попыток stop именованной unit, пауза 3 с; затем state сохраняется при отказе |
| Управляющий stop | До 75 с ожидания owner, затем generation-fenced cleanup; зависший WSL broker может задержать синхронный WSL вызов сверх этого времени |
| Backend stall detector | Подготовленный профиль задаёт 300 с без progress; реальные worker fault tests ещё отдельный этап |

`reset-failed` существует только в явном ручном start/restart. Supervisor не сбрасывает systemd limit циклически. Нельзя обходить исчерпание лимита бесконечными ручными рестартами: сначала сохранить numeric receipt и причину отказа.

Job гарантирует lifetime Windows-процессов; Linux cgroup убирает внешний watcher. Прежний `sleep infinity` заменён bounded `keepalive.py`: атомарный heartbeat supervisor каждые 5 с (запись повторяется до 20×50 мс; сбой heartbeat терпится, пока последняя успешная запись моложе 20 с), выход при подтверждённом удалении state (два промаха через ≥0,2 с)/смене owner/lease либо 30 с без роста sequence. Новый lease id при каждом Keepalive restart завершает и прежний helper того же owner. Защищённый receipt содержит generation, lease, Linux PID и `/proc` starttime. Четыре own Linux helper cases прошли; интеграция нового Windows supervisor + реальный production helper остаётся пунктом живой приёмки. Точные read-only команды до/после crash — [keepalive-review.md](keepalive-review.md). Одновременная смерть owner и watcher, общий ancestor job, отказ всего Windows/WSL и cold boot не покрываются собственными synthetic children.

Автозапуск при загрузке или входе Windows не добавляется. До ручного запуска владельцем сервера нет. Sleep request принадлежит потоку supervisor, не меняет power plan и исчезает с ним. Это не обещание доступности при выключенном ПК.

## Установка и rollback

Все следующие команды — инструкции владельцу релиза. Они **не выполнялись при подготовке этого документа**. Нельзя одновременно запускать lifecycle/fault tests, пользовательскую генерацию, experiment или другой контроллер. Сначала завершить собственную GPU-квалификацию, зафиксировать исходники/библиотеки и квалифицированные профили. Не менять `qualified` лишь ради обхода installer.

В PowerShell из корня workspace:

```powershell
Set-Location 'C:\path\to\qwen36-workspace'
# Только проверка Windows source/parser; без WSL и сервисов.
powershell.exe -NoProfile -File outputs/qwen36-api/optimization/release-management.ps1 -CheckOnly

# Далее только после предоставления окна обслуживания.
qwen36 stop
# Если владелец завершил именно наш isolated experiment:
wsl.exe -d archlinux -- systemctl --user stop qwen36-experiment.service
wsl.exe -d archlinux -- systemctl --user show qwen36-inference.service qwen36-experiment.service --property=Id --property=LoadState --property=ActiveState --property=SubState
```

Обе units должны быть подтверждённо inactive/failed либо `LoadState=not-found, ActiveState=inactive, SubState=dead`. Неизвестное состояние, ошибка systemctl и activating/deactivating запрещают установку. Наличие `run\supervisor.json` также блокирует Windows installer; файл не удалять вручную для обхода.

Linux release команды внутри существующего WSL от USER:

```bash
cd /path/to/qwen36-workspace
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/release.py install --check
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/release.py install
```

Linux installer требует квалифицированный default, проверяет обе units и печатает `rollback_manifest`. Сохранить именно этот путь. После успешного Linux release, в PowerShell:

```powershell
powershell.exe -NoProfile -File outputs/qwen36-api/optimization/release-management.ps1
```

Windows installer блокирует оба control mutex, проверяет обе units, сохраняет защищённую полную копию своих пяти файлов и печатает отдельный путь rollback manifest. Он не запускает модель. Это два отдельных receipts; сбой второго шага не означает автоматический откат первого. После установки нужно проверить их хэши и выполнить живую приёмку. Файловый release не является общей транзакцией сразу Windows+Linux; при I/O-сбое возможен частичный release с заранее подготовленным manifest.

Для rollback выбрать **точные manifests напечатанного релиза**, а не «последнюю папку». Значения ниже — переменные владельца, не реальные пути выданных receipts:

```powershell
# $windowsManifest содержит точный путь под C:\path\to\LLM\qwen36-api\backups.
powershell.exe -NoProfile -File outputs/qwen36-api/optimization/release-management.ps1 -RollbackManifest $windowsManifest -CheckOnly
powershell.exe -NoProfile -File outputs/qwen36-api/optimization/release-management.ps1 -RollbackManifest $windowsManifest
```

```bash
# linux_manifest содержит точный /home/USER/qwen36-api/backups/.../manifest.json.
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/release.py rollback --manifest "$linux_manifest" --check
# Обе собственные units предварительно остановлены.
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/release.py rollback --manifest "$linux_manifest"
```

Windows rollback явно выполняет штатный stop и оставляет сервис выключенным; Linux rollback требует уже остановленные units. Оба заранее проверяют все destination и backup хэши. Более поздняя правка или повреждённая копия запрещает rollback до первой замены; не удалять чужие правки, чтобы «прошла проверка». Повторная проверка перед заменами не защищает от произвольного параллельного редактора: на всё окно выпуска требуется эксклюзивность. Откат старого supervisor возвращает также прежние ограничения его ownership.

## Живая приёмка владельцем

Следующие тесты действительно останавливают/перезапускают **свой** сервис; запускать только после release и выделения окна. Полный lifecycle начинает со stopped production+experiment и заканчивает stopped. Пять файлов controller/launcher должны соответствовать staged версии; четыре PowerShell controller файла сравниваются harness по хэшам.

```powershell
# Без -Live: parser/план; без запуска модели.
powershell.exe -NoProfile -File outputs/qwen36-api/optimization/test-ops-v2.ps1

# Ручные start/status/repeated-start/restart/stale token/stop,
# own tunnel+metrics crash, затем own supervisor crash и watcher cleanup.
powershell.exe -NoProfile -File outputs/qwen36-api/optimization/test-ops-v2.ps1 -Live -IncludeFaults -Profile main128 -ReadyTimeoutSeconds 600
```

Harness сохраняет конкретные owner generations, держит process handles и повторяет PID+creation-time проверку перед Kill. Он не завершает WSL, не ищет процессы по общему имени и отказывается при смене owner. При сбое tunnel модель обязана остаться той же; если краткий outage успел восстановиться до наблюдения, receipt содержит `NOT_OBSERVED`, а не утверждение проверенного отсутствия связности. После owner crash watcher должен убрать state/Windows children и named unit; 15 с без воскресшего сервиса и новый явный start проверяются отдельно. Реальное освобождение GPU и исчезновение Linux Keepalive дополнительно фиксирует владелец приёмки.

После успешного lifecycle заново запустить профиль и дождаться status=ready, затем из WSL выполнить:

```bash
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/network-acceptance.py
```

Этот script **live при вызове**, у него нет dry-run. Ключ идёт только через SSH stdin на существующий `EDGE_SSH_HOST`. Проверяются authenticated models, nonstream и SSE `42`, alias, DONE; отсутствие/неверный ключ должны давать **401**, а не 403/429/502/transport error. Закрытые маршруты должны давать 404; старый forge — 200. Неподдерживаемые запросы (forced `tool_choice`, `response_format` json, `parallel_tool_calls:false` при tools, не-bool `parallel_tool_calls`, `strict:true` на function или tool, legacy `functions`) должны давать 400 JSON-ошибкой до SSE в stream и nonstream, на loopback и HTTPS. Отчёт привязан к одному instance (`instance_id` до и после, `same_instance`). HTTPS с проверкой сертификата, запретом redirects и inherited proxies. Возврат успешного DNS/TLS без generation недостаточен. Это Caddy-host/LAN тест, `external_e2e=NOT_RUN` сохраняется.

Worker recovery проверяется **отдельно**, после нового ручного старта и readiness, без ops harness и чужих запросов:

```bash
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/test-worker-fault-v2.py --live --case kill
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/test-worker-fault-v2.py --live --case stall
```

Не запускать оба одновременно. Скрипт выбирает единственного подтверждённого multiprocessing worker только из cgroup нашей unit, сверяет `/proc` starttime, executable, ancestry и свежие placement receipts, затем сигналит через pidfd. SIGSTOP имеет независимый resume guard 380 с; ожидается runtime stall shutdown после 300 с, новый systemd invocation и короткая локальная генерация. Подробные ограничения и возврат SIGCONT — [worker-fault-v2-acceptance.md](worker-fault-v2-acceptance.md). Скрипт не выполняет start/stop/reset-failed. Оба сбоя расходуют штатный restart budget; не ослаблять его ради PASS.

Если тест отказал, сохранить его receipt и `qwen36 status`. Для своей подтверждённой ручной сессии использовать `qwen36 stop`, затем адресный rollback или исправление. При обнаружении нового owner harness сохраняет его: не удалять state и не kill по имени. Если WSL broker завис, timeout не разрешает `wsl --terminate` или перезагрузку. Для прерывания worker harness использовать Ctrl+C/SIGINT его процесса: finally и независимый guard адресно посылают SIGCONT; при убийстве всей среды такая гарантия отсутствует.

## Метрики, приватность и границы покрытия

Владелец WSL хранит `logs`/reports с закрытыми правами. HTTP метрики пишутся в bounded queue отдельным writer; переполнение учитывается `metrics_dropped`. NVML семплируется фоном, а не синхронно на каждом запросе. Engine/request rows соединяются по `(run_id, engine_uid)`; UID без run_id недостаточен, поскольку процесс сбрасывает счётчик. Partial JSONL не считается готовой строкой. Поздно записанные engine данные ожидаются ограниченное время; отсутствие записи отмечается, не заменяется строкой старого процесса.

Сохраняются числовые времена, token counts, request-local sampling, cache/placement/transfer counters и обезличенные исходы. **Не сохраняются** prompts, ответы, tool arguments, Authorization, альтернативные ключевые заголовки и произвольные error strings. Request ID/instance ID служат связыванию метрик. Payload synthetic tests также не печатается в результаты; сохранённый hash текста не является оценкой качества модели. Resource peaks семплированные; VRAM всей GPU включает другие приложения, память WSL VM отличается от service RAM. Raw Nsight/engine diagnostics не помещать в публичный отчёт без фильтрации.

| Покрытие этого этапа | Статус |
|---|---|
| Win32 job: normal/crash/nested/unpublished child/stale watcher/fail-before-model | 7 сценариев с собственными реальными Windows children PASS; [receipt](lifetime-job-results.json) |
| Controller identity/restart/argv/heartbeat/rollback guards | 39 mock/static checks PASS, без service/GPU |
| Linux bounded keepalive | 15 CPU tests и 4 реальных own helper cases PASS; [keepalive-live-results.json](keepalive-live-results.json); production Job integration НЕ ПРОВЕРЕНА |
| Release preflight/повреждённый поздний backup | CPU fixture tests; [release-safety-results.json](release-safety-results.json) |
| Readiness redirects/proxies/TLS/identity | 7 transport tests и 11 existing mocks PASS; [readiness-security-results.json](readiness-security-results.json) |
| Ops harness | 18 CPU mocks PASS; полный live service lifecycle НЕ ЗАПУЩЕН этим этапом |
| Worker/network harness | 16 и 14 CPU tests PASS; новый live SIGSTOP/SIGKILL и Caddy generation НЕ ЗАПУЩЕНЫ этим этапом |
| Новый management release и Linux cleanup | До receipts владельца релиза НЕ ПРОВЕРЕНО |
| Cold boot Windows / до интерактивного входа | Перезагрузка не выполнялась, автозапуска нет |
| Генерация с отдельной внешней машины | НЕ ПРОВЕРЕНО; Arch недоступен исполнителю |

Из исходного [README](../README.md) сохраняются сетевые ограничения: Caddy находится на домашнем `EDGE_SSH_HOST`; публичное имя проверялось из WSL/edge, это не внешняя точка. Нативный Windows доступ по публичному имени/IP ранее таймаутил также для старого forge, при LAN+SNI проверка проходила. Причина Windows route, DHCP reservation edge, WAN статичность, NAT/DDNS и router credentials не установлены. Этот этап их не меняет и не объявляет исправленными. Для реального внешнего теста владелец использует [test-external.sh](../test-external.sh) с Arch через мобильный интернет после ручного запуска сервера; API-key нельзя отправлять сторонним сервисам проверки.
