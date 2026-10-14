# Экспорт результатов и исходных патчей

Подготовлен [export-release-evidence.py](export-release-evidence.py). **Реальный release export не выполнялся** по поручению владельца текущего run. Выполнены десять [CPU-тестов](test-export-release-evidence-cpu.py) на отдельных временных Git/JSON fixtures: reverse-check, применение в чистую synthetic baseline и побайтовое восстановление, обезличивание и отказы. Никакие текущие model/results/overlay данные для тестов не читались. Runtime, management, GPU, сервисы, рабочий Git index/HEAD и значения `qualified` не менялись.

## После заморозки исходников и результатов

Exporter использует стандартную библиотеку Python и локальный Git. Рекомендуемый запуск — внутри уже существующего WSL, где доступны выбранные private numeric JSON. Ничего устанавливать или запускать на GPU он не будет.

```bash
cd /path/to/qwen36-workspace
/home/USER/qwen36-api/.venv/bin/python outputs/qwen36-api/optimization/export-release-evidence.py --name release-evidence
```

Это создаст **новый** каталог `outputs/qwen36-api/optimization/evidence-exports/release-evidence`. Уже существующий export не перезаписывается. По умолчанию выбираются перечисленные в `DEFAULT_RESULTS` публичные numeric component/CPU JSON и profiles snapshot. Отсутствующие необязательные public reports перечисляются в manifest, а не считаются пройденными.

В итоговый запуск явно добавить **фактические имена** завершённых benchmark/quality/soak/network JSON:

```text
--result native:reports/<фактический-result>.json
--result native:active-run.json
```

Повторять `--result` для каждого выбранного файла. Угловые скобки — обозначение места для фактического имени, не команда для копирования. Имя не должно содержать пути вне разрешённого дерева. Явно выбранный отсутствующий файл — ошибка. `--native-root` меняет корень native optimization, по умолчанию `/home/USER/qwen36-api/optimization`. Из native разрешены только отдельные `reports/...json` и `active-run.json`; ни log-каталоги, ни secret/config env не читаются.

`--no-default-results` позволяет экспортировать только явно выбранные JSON. Запуск на Windows допустим для public-only экспорта; native WSL root тогда не используется, если нет соответствующих selectors. Сам скрипт никогда не вызывает `wsl.exe`.

## Структура результата

```text
manifest.json
README.md
runtime/integration.patch
runtime/features/A.json
runtime/features/B.json
runtime/features/C.json
runtime/features/D.json
runtime/features/E.json
service-overlay/...
evidence/public/...
evidence/native/...
evidence/service/profiles.json
```

**Runtime baseline:** точный FreeToken commit `cac247a860e316e06580d05aeb05f2e647bde214`. Exporter откажется работать, если HEAD уже отличается: это требует отдельного review базы, а не молчаливой смены точки сравнения. Checkpoint/revision и donor Strata revision записываются как provenance; веса не читаются и не копируются.

В `integration.patch` входят изменённые tracked source files и новые untracked `.py/.cu/.cpp/.h` и родственные source extensions из `python/freetoken/` и `tests/`. Git diff строится с отключёнными external diff/textconv и без renames. Новые файлы добавляются как обычные new-file patch hunks; Git index не меняется. Неизвестный изменённый tracked файл вне scope или source без группы вызывает отказ, чтобы пакет не выглядел полным при пропущенном изменении.

Реальные изменения пересекаются в engine, scheduler, model, attention, MoE cache. Поэтому **отдельных якобы независимо применимых A–E patches нет**. Один общий patch плюс per-feature manifests честно фиксируют:

- **A:** telemetry, API, sampler/request seed, scheduler correctness.
- **B:** CUDA kernels, NVFP4 prefill, конфигурация.
- **C:** residency/complement и транспорт экспертов.
- **D:** KV, GDN state, prefix snapshots.
- **E:** speculative decoding/MTP/lookup.

У каждой группы есть пути, SHA-256, shared dependencies, наблюдавшиеся имена env switches и пояснение отключения. Эти записи — описание механизма, **не acceptance**. Security/correctness fixes не имеют безопасного «off» флага; для них нужен review соответствующего release rollback. Опциональные CUDA/FP8/complement/speculation контролируются своими переключателями и qualified profile loader. Наличие исходника или переключателя не включает feature и не повышает её статус.

Root service scripts (guard/ingress/metrics/readiness/profile runtime/manual supervisor и launcher) копируются точными байтами в отдельный **service overlay**, без upstream patch. `profiles.json` сохраняется без изменения `qualified`. Связанный script `get-api-key.ps1` является кодом чтения уже существующего защищённого ключа; сам файл ключа не копируется. Bundle ничего не устанавливает. Build recipes для отдельных CUDA component suites остаются в соседних исходных deliverables `cuda-nvfp4`, `cuda-kv`, `cuda-transport` и `cuda-layer-fusion`; экспорт не заменяет их полного окружения сборки и не содержит `.so`.

## Обезличивание evidence JSON

Это намеренно **lossy numeric export**, а не копирование JSON без проверки. Рекурсивно удаляются prompts/messages/ответы/tool args, token-ID массивы, headers, credentials, stdout/stderr/test-output, свободные descriptions/notes, process/session/request/run identifiers, timestamps и absolute location strings. Сохраняются числа, boolean/null, hashes, package versions и ограниченные metric/status/case/profile labels. Числовые строки профайлера преобразуются в числа; kernel names и единицы метрик сохраняются в разрешённой схеме. Неизвестные свободные строки удаляются.

Manifest содержит SHA-256 исходного выбранного JSON, SHA-256 обезличенного JSON и счётчики удаления по причинам. Path keys в file hash maps сокращаются до basename; при коллизии exporter отказывается продолжать. Исходные JSON не редактируются. Нельзя использовать sanitized JSON для побайтового восстановления старого исходного отчёта: за него отвечает `original_sha256` и сохранённый private original.

Для component evidence явно указывается `synthetic_component_not_model_speed`, для известных CPU suites — `cpu_mock_or_static_validation_not_live_service`. Остальным exporter не назначает qualification scope автоматически. Он не вычисляет новые A/B выводы и не меняет исходные pass/fail/qualified boolean. Если sanitization удалил важный свободный caveat, owner сохраняет его в итоговом отчёте отдельно; компонентное ускорение не становится end-to-end выигрышем.

Исходники в patch и service overlay сохраняются **точными байтами**, включая deployment-specific пути и публичное имя хоста, если они есть в коде. Это отдельное свойство source bundle, не обещание его полной анонимности. Перед внешней публикацией требуется просмотр владельцем; сейчас output локальный. Проверка ищет PEM private keys и подозрительные literal credentials; env-файлы, секреты и raw logs вообще не входят в allowlist.

## Проверки exporter при будущем запуске

- Новый каталог назначения без overwrite; сначала `.preparing-*`, затем rename после всех проверок.
- Ограниченные text-source/JSON размеры, UTF-8, отсутствие NUL/binary patches, запрет symlinks/path escape.
- `git apply --check --reverse` проверяет integration.patch против текущего изменённого worktree, **не применяя** его.
- Повторный SHA-256 source/overlay и повторное чтение inventory/HEAD запрещают завершить export из смешанного меняющегося snapshot.
- При отказе `.preparing-*` остаётся для диагностики; скрипт ничего рекурсивно не удаляет.

До реального применения в отдельном чистом checkout закреплённого baseline выполнить `git -c core.autocrlf=false apply --check <export>/runtime/integration.patch`; при применении также передать `-c core.autocrlf=false`. Windows Git с глобальным autocrlf иначе меняет окончания строк даже при корректном patch. Для реального runtime это ещё **не выполнено**; synthetic fixture уже проверяет modified/deleted/new/empty/no-final-newline/CRLF файлы. Reverse-check существующего worktree не заменяет функциональные тесты чистой установки или реальный GPU benchmark. Exporter не создаёт checkout, не компилирует kernels, не запускает inference, не управляет DNS/Caddy/firewall, не перезапускает службы и не делает commit/push.

Проверка credential literals распространяется и на удалённые строки patch. Чувствительные `key/env/environment` поля удаляются; строка из 40/64 hex-символов сохраняется как hash только с явным hash/revision/commit именем либо в file-hash map. Запрет raw/log/secret selectors проверяется до чтения, включая регистр имён. Обезличивание остаётся схемной защитой для выбранных numeric artifacts, а не гарантией безопасной публикации произвольного чужого JSON.

Источник qualification остаётся у владельца релиза: реальные model A/B, качество, boundaries, soak и ops/network acceptance. Этот пакет лишь сохраняет уже выбранные доказательства и текущие исходники.
