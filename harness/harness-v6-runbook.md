# Harness V6 — runbook для родителя (R1: FP32 NVFP4-global, CUDA bit-exact, final4-r1)

Статус подготовки: только CPU. Модель, GPU и службы не запускались, в native ничего не записано.
final3-v5 остаётся INCONCLUSIVE как запись истории и не перемаркируется.

## 0. Что изменилось в V6 (кратко)

- **compare-ab.py V6.** Гейт считается только по полному времени запроса (elapsed). Гейтовых кейсов 9: fresh 64/4096/16384/65536/120000, history_continuation, last_message_edit, tool_result_edit (`elapsed_s`) и tool_cycle (`task_elapsed_s`). Метрики `ttft_content_s` и `decode_client_tps` только выводятся в отчёт.
  - Статистика: t-распределение (df=4) по log-отношениям ровно 5 пар.
  - Non-inferiority: односторонняя нижняя граница 97.5% ≥ 1/1.03 для **каждого** кейса.
  - FAIL: верхняя граница 97.5% < 1/1.03 хотя бы у одного кейса. Сохранено и правило V5: если верхняя граница < 1.05 у всех кейсов, «gain excluded» тоже даёт FAIL.
  - Gain: нижняя граница с поправкой Бонферрони ≥ 1.05 хотя бы у одного кейса. Поправка F = 9 кейсов × (число профилей − 1). Для final4-r1 F = 9×3 = 27, t(1−.025/27, 4) = 7.3214.
  - Метка `gain_not_established` ставится только тогда, когда gain действительно не установлен.
  - Exit-коды:

    | Код | Значение |
    |---|---|
    | 0 | PASS |
    | 1 | FAIL |
    | 2 | INCONCLUSIVE |
    | 3 | INVALID |
    | 4 | внутренняя ошибка или ошибка usage, либо файл сравнения уже существует (вердикта нет) |
    | 75 | GPU-lock занят |

  - Отчёт `reports/NAME-CAND-vs-BASE-comparison.json` создаётся с O_EXCL и никогда не перезаписывается.
  - Проверка native↔manifest (`current_run_spec`, включая `launcher_files`) выполняется под неблокирующим GPU-lock.
- **bench.py / matrix.py V6.**
  - Manifest содержит `launcher_files`: sha256 файлов start-experiment.sh, run-experiment.sh, check-experiment.py. Там же `vram_floor_bytes`.
  - Перед каждым arm заново хэшируется вся identity. Конфиг в workspace должен совпадать с native.
  - После readiness проверяется identity лаунчера: из `/proc/<MainPID>/environ` берутся фильтрованные переменные, секреты редактируются. Сверяется с manifest, со статическим контрактом run-experiment.sh и с первым arm (`NAME-launcher.json`, O_EXCL). Эта же identity пишется в bench-отчёт как `launcher`.
  - В строке каждого arm записываются:
    - чужая VRAM до старта (nvidia-smi, без CUDA-контекста);
    - минимум свободной памяти GPU, который измеряет сервис.
  - Если свободной памяти < 0.5 GiB, arm получает INVALID с кодом `gpu_min_free_below_floor`; матрица доходит до конца и завершается с exit 3.
  - Exit-коды matrix:

    | Код | Значение |
    |---|---|
    | 0 | всё в порядке |
    | 1 | сбой arm |
    | 2 | отказ до старта |
    | 3 | есть INVALID arm |
    | 75 | lock занят |

  - `state()` работает в режиме fail-closed. Если старт упал, осиротевший unit останавливается. Дрейф identity посреди матрицы записывается строкой.
- **start-experiment.sh (P-1/P-2).**
  - Копирования всех `*.sh` и `*.json` из workspace больше нет.
  - Копируется только конфиг запускаемого профиля по точному имени: проверка `^[a-z0-9][a-z0-9._-]{0,120}$`, atomic `mv`, режим 600.
  - Native-only конфиг допускается.
  - `profile-final-*` запускается через profile.sh.
  - Поток teacher-`<cand>` в qualify.py продолжает работать: проверено тестом.
- **Конфиги CUDA bit-exact** (новые; существующие JSON не менялись):

  | Файл | chunk | ratio | Назначение |
  |---|---|---|---|
  | `cuda-bitexact128-2048.json` | 2048 | .75 | arm 3 матрицы: main-retain2048 + 2 env |
  | `cuda-bitexact128.json` | 4096 | .73 | arm 4 матрицы: main-retain + 2 env |
  | `cuda-bitexact128-c2048.json` | 2048 | .75 | screening; байт-в-байт по смыслу = `cuda-bitexact128-2048` |
  | `cuda-bitexact128-c6144.json` | 6144 | .71 | screening |
  | `cuda-bitexact128-c8192.json` | 8192 | .69 | screening |
  | `cuda-bitexact128-c16384.json` | 16384 | .63 | screening |
  | `teacher-cuda-bitexact128{,-2048}.json` | — | — | те же байты, которые пишет qualify.py |

  - 2 env: `QWEN36_NVFP4_PREFILL=cuda` и `QWEN36_CUDA_NVFP4_LIBRARY=/home/USER/qwen36-api/optimization/private-libraries/libqwen_nvfp4_bitexact.so`.
  - Подбор ratio описан в §6.
- **sync-v5.sh.**
  - Allowlist расширен. Добавлены: screen-chunk.py, compare-logit-reference.py, test-tool-choice-guard-cpu.py, test-tool-semantics-guard-cpu.py, test-soak-cpu.py, 8 CUDA-конфигов. Группа `lib`: `work/cuda-nvfp4-bitexact/build/libqwen_nvfp4_bitexact.so` → `private-libraries/` (файл 0600, каталог 0700).
  - **sha256 библиотеки нигде не захардкожен, его фиксирует только `--freeze`.** Пересобранная .so получает новый freeze. Если .so изменилась после freeze, apply завершается с 65. Если .so отсутствует, это блокер 66.
  - Runtime tree синхронизируется в режиме по умолчанию, то есть без `--skip-runtime`: это проверено sandbox-тестом T21. Native-only `.so` в `python/freetoken/kernel` сохраняются, но **не пересобираются**.
  - CPU-checks.diff (guard v2 suites) применён к CPU-checks.sh.
- **Новые скрипты.** screen-chunk.py (screening чанка, не acceptance) и compare-logit-reference.py (информационное сравнение старого и нового teacher-эталона).

## 1. Переменные (WSL, пользователь USER)

```bash
W=/path/to/qwen36-workspace; O=$W/outputs/qwen36-api/optimization
N=/home/USER/qwen36-api/optimization; PY=/home/USER/qwen36-api/.venv/bin/python
LOCK=/home/USER/qwen36-api/run/gpu-exclusive.lock
```

Как держат GPU-lock:
- Сами (обёртку `flock` **не** использовать): `sync-v5.sh --apply`, `matrix.py`, `tool-pilot.py`, `screen-chunk.py`, `compare-ab.py`.
- Только под `flock -n $LOCK`: `qualify.py` и однострочник functional.
- Всё, что идёт дольше 8 минут, запускать в фоне (`run_in_background`) и следить через Monitor.

## 2. Предусловия (STOP, если не выполнены)

1. Агент fp32-global завершил работу и опубликовал:
   - хэши изменённых файлов `work/FreeToken-opt`;
   - sha256 пересобранной `work/cuda-nvfp4-bitexact/build/libqwen_nvfp4_bitexact.so`;
   - нужна ли пересборка native-only `.so` (`kernel/_cpu_moe*`, `_pinned_tensor*`, `_ple_store*`). sync эти файлы не пересобирает.
2. Все фоновые агенты остановились. Units `qwen36-inference` и `qwen36-experiment` в состоянии inactive. GPU-lock свободен.

## 3. Шаги по порядку

### Шаг 1. Freeze R1 (только чтение, ~1 мин)

```bash
bash $O/sync-v5.sh --print-freeze > $O/sync-r1-freeze.sha256          # БЕЗ --skip-runtime
grep -c '  work/FreeToken-opt/python/freetoken/' $O/sync-r1-freeze.sha256   # > 0 (≈457)
grep '  work/cuda-nvfp4-bitexact/build/libqwen_nvfp4_bitexact.so$' $O/sync-r1-freeze.sha256  # = sha от fp32-global
python3 $W/work/agents/harness-v6/make-freeze.py --check $O/sync-r1-freeze.sha256   # exit 0: 0 frozen mismatch
```

Пояснения к последней команде:
- Если `--check` показывает MISMATCH, значит файл из `harness-v6-freeze.json` изменился после V6. STOP: нужна переоценка.
- Строки MUTABLE и RUNTIME выводятся только для информации, это и есть R1-значения.

### Шаг 2. Бэкап истории эталона V5 в native (~1 мин; до любого teacher-прогона R1)

Режим `teacher-bf128` перезаписывает `reports/logit-reference/*.npy` на месте. `quality.py` перезаписывает `reports/NAME.json`.

```bash
[ ! -e $N/reports/logit-reference-pre-r1 ] && cp -a $N/reports/logit-reference $N/reports/logit-reference-pre-r1
for f in teacher-bf128-logits functional-bf128 teacher-main-retain2048 functional-main-retain2048 quality-main-retain2048; do
  [ -f $N/reports/$f.json ] && cp -n $N/reports/$f.json $N/reports/$f-pre-r1.json; done
```

### Шаг 3. Sync R1 (~3 мин) и CPU-проверки в native (~2 мин)

```bash
bash $O/sync-v5.sh --dry-run --label r1 --freeze $O/sync-r1-freeze.sha256   # rc 0, "freeze ...: 0 mismatch(es)"
bash $O/sync-v5.sh --apply   --label r1 --freeze $O/sync-r1-freeze.sha256   # rc 0, APPLY PASS
bash $O/sync-v5.sh --verify  --label r1                                      # 0 drift(s)
cd $N && CUDA_VISIBLE_DEVICES= $PY test-paired-contract-cpu.py | tail -1   # PASS, 57 tests, 25/25 mutants
cd $N && CUDA_VISIBLE_DEVICES= $PY test-soak-cpu.py 2>&1 | tail -2        # OK (79)
QWEN36_GUARD_UNDER_TEST=$N/guard.py bash $O/CPU-checks.sh > /tmp/cpu-r1.log 2>&1; echo rc=$?   # 0
```

Если fp32-global потребовал пересборку native-only `.so`, её нужно выполнить по его инструкции **до шага 4**. sync этого не делает.

### Шаг 4. Новый эталон bf128-R1 (~25–30 мин GPU)

4a. Functional 5/5 на bf128-R1, ~7 мин. Отчёт пишется под стандартным именем `functional-bf128`, поэтому следующие qualify берут его как functional-baseline:

```bash
flock -n $LOCK bash -c 'set -e; ! systemctl --user is-active --quiet qwen36-experiment || exit 73
  trap "systemctl --user stop qwen36-experiment" EXIT
  bash '$N'/start-experiment.sh bf128; '$PY' '$N'/check-experiment.py 300
  cd '$N' && '$PY' '$N'/quality.py --name functional-bf128'
```

4b. Teacher-эталон bf128-R1 и повторная квалификация main-retain2048 на R1, ~15–20 мин. Профиль `teacher-bf128` записывает новый `reports/logit-reference`:

```bash
flock -n $LOCK $PY $N/qualify.py --candidate main-retain2048 --baseline-teacher
```

Ожидается в `reports/quality-main-retain2048.json`:
- `logit_comparisons.expected == observed == 125`, `max_abs == 0`;
- `forced_targets_complete=true`;
- `new_functional_failures=[]`;
- `ppl_gate_pass=true`.

4c. Насколько сдвинулся численный базис. Это только информация, не гейт:

```bash
$PY $N/compare-logit-reference.py --name logit-reference-r1-vs-pre-r1
```

По умолчанию сравниваются `reports/logit-reference-pre-r1` и `reports/logit-reference`, а также два последних run_id `teacher-bf128` в `teacher.jsonl`. В отчёте:
- max_abs и relative_l2 логитов;
- согласие top-1;
- относительное изменение perplexity.

Ненулевой diff здесь ожидаем, потому что базис изменился. Если файлов `only_old` или `only_new` больше 0, нужно разобраться.

### Шаг 5. Teacher и functional CUDA-кандидатов против эталона bf128-R1 (~25 мин GPU)

```bash
flock -n $LOCK $PY $N/qualify.py --candidate cuda-bitexact128-2048
flock -n $LOCK $PY $N/qualify.py --candidate cuda-bitexact128
$PY $N/check-candidate-metrics.py --candidate cuda-bitexact128-2048; $PY $N/check-candidate-metrics.py --candidate cuda-bitexact128
```

Ожидается для обоих: logits `max_abs == 0` против bf128-R1 (bit-exact), `forced_targets_complete`, `new_functional_failures=[]`. Любой ненулевой diff даёт FAIL до разбора: тогда этот профиль исключается из матрицы.

### Шаг 6. Screening размера чанка, CUDA bit-exact (~20–25 мин GPU, 1 повтор; не acceptance)

```bash
$PY $N/screen-chunk.py --name final4-r1-screen      # профили по умолчанию: c2048, c4096(cuda-bitexact128), c6144, c8192, c16384
```

Опции: `--profiles ...`; чтобы добавить Triton-якорь, укажите `main-retain2048` в начале списка, это +4 мин.

Что пишется:
- `reports/final4-r1-screen-<profile>-chunk-screen.json` по каждому профилю;
- сводка `reports/final4-r1-screen-chunk-screen.json`.

Метрики: TTFT, elapsed, prefill tok/s (клиентский и engine `prefill_gpu_ms`), decode tok/s, min GPU free, `expert_counter_deltas.prefill_h2d_bytes`/`decode_missing` (логические байты), expert_slots.

Exit-коды:

| Код | Значение |
|---|---|
| 0 | всё измерено |
| 1 | какой-то профиль упал; остальные измеряются дальше |
| 2 | отказ |
| 75 | lock занят |

**Заранее объявленное правило выбора 4-го плеча.** По умолчанию arm 4 = `cuda-bitexact128` (c4096). Его заменяют на c6144 или c8192, только если выполнены все условия:
- (a) status measured и vram_gate PASS (≥ 0.5 GiB);
- (b) elapsed на 120000 хотя бы на 3% ниже, чем у c4096;
- (c) elapsed на 16384 и на 65536 не выше, чем у c4096;
- (d) decode_client_tps ≥ 0.97 × c4096.

c16384 только для screening: валидатор production допускает чанк 1024–8192, слотов на −33%. Решение по нему принимается отдельно и позже. Если 4-е плечо заменено, его нужно квалифицировать (шаг 5 с новым именем). В матрицу без qualify не идёт ни один профиль. Также нужно поменять `--profiles` в шаге 8. Семейство остаётся равным 3.

### Шаг 7. Tool pilot V6 на bf128 (~4 мин GPU)

```bash
$PY $N/tool-pilot.py --profile bf128 --repeats 3 --name final4-r1-toolpilot-bf128   # exit 0 = PASS 3/3
```

### Шаг 8. Подтверждающая матрица final4-r1 (~85–100 мин GPU, в фоне)

```bash
$PY $N/matrix.py --profiles bf128,main-retain2048,cuda-bitexact128-2048,cuda-bitexact128 --pairs 5 \
  --name final4-r1 --output 128 --tool-pilot final4-r1-toolpilot-bf128
```

Длины по умолчанию: 64,4096,16384,65536,120000. Кейсы те же, что в final3-v5.

Матрица: 20 arm, сбалансированное расписание. Exit-коды: 0, 1 (сбой arm, resume только для preflight/start/readiness), 2, 3 (есть INVALID arm по VRAM: смотреть `invalid_arms`), 75.

### Шаг 9. Compare V6, семейство кандидатов = 3 (<1 мин каждый, без обёртки flock)

```bash
$PY $N/compare-ab.py --name final4-r1 --candidate main-retain2048       --baseline bf128
$PY $N/compare-ab.py --name final4-r1 --candidate cuda-bitexact128-2048 --baseline main-retain2048
$PY $N/compare-ab.py --name final4-r1 --candidate cuda-bitexact128      --baseline main-retain2048
```

Exit-коды: 0 PASS, 1 FAIL, 2 INCONCLUSIVE, 3 INVALID, 4 (нет вердикта: ошибка или отчёт уже есть), 75 (lock занят).

Повторный запуск под тем же `--name`/кандидатом/baseline отклоняется с exit 4. Для повтора нужно новое имя матрицы.

В отчёте:
- `gating_cases` (9 штук);
- `rows` со строками `gating=false` для ttft/decode;
- `tool_task`;
- `gain_comparison_family_size` = 27;
- `vram_observations`, `launcher_environment_sha256`;
- `drift_keys`, если native изменился.

## 4. Оценка длительности

GPU-шаги:

| Шаг | Длительность |
|---|---|
| 4 | ~30 мин |
| 5 | ~25 мин |
| 6 | ~25 мин |
| 7 | ~4 мин |
| 8 | ~85–100 мин (final3-v5: 4.2 мин/arm × 20) |
| 9 | ~3 мин |

Итого ≈ 2.9–3.2 ч GPU плюс ~10 мин CPU (шаги 1–3). Если 4-е плечо заменено после screening, добавляется ещё ~12 мин qualify.

## 5. Не выполнено (NOT_RUN) и риски

- NOT_RUN: все шаги 3–9 на GPU, `sync-v5 --apply` в native, live-проверки.
- Активации для c6144 и c16384 оценены интерполяцией и экстраполяцией, они не измерены. Прогноз c16384/.63: 0.61–0.89 GiB, 2112 слотов, ожидается заметная потеря decode. Кратковременный всплеск чужой VRAM (+0.95 GiB) может пробить порог при любом ratio.
- Окружение лаунчера читается из `/proc/<MainPID>/environ`, то есть это стартовое окружение процесса. Изменения, которые experiment.py вносит в os.environ после старта (конфиг-env, ключ), в отчёт не попадают, а ключ не попадает никогда. Если на native-хосте окружение user-manager отличается от ожидаемого статического контракта (PYTHONPATH/CUDA_HOME/LD_LIBRARY_PATH/без LD_PRELOAD), arm получает failed_step `launcher`.
- `test-release-qualification-cpu.py`: 35/36 при записи `.pyc`, 36/36 с `PYTHONDONTWRITEBYTECODE=1`, как его запускает сам release-gate. Это артефакт теста: в снимок temp-root попадает `__pycache__`. С harness не связано.
- Если fp32-global изменил исходники native-only `.so`, без пересборки такие `.so` будут старыми. sync-v5 их не трогает.
