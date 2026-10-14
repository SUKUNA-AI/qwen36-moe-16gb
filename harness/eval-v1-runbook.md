# eval-v1: выбор базиса NVFP4 global (fp32 или fp16-compat), runbook для родителя

Агент eval-suite, 2026-10-14. Всё CPU-часть выполнена агентом. GPU-прогоны (шаги B–C) запускает родитель.

## 0. Что сделано и зачем

На эталоне bf128 переход fp16 → fp32 global дал functional 3/5. Упали executable-code и long-facts-16384.
5 кейсов слишком мало, а именно эти два чувствительны к любому малому сдвигу чисел. Поэтому сделан
набор из 60 кейсов, а правило решения зафиксировано до прогона.

| Файл | sha256 |
|---|---|
| `eval-v1/freeze.json` (кейсы, профили, runtime, правило) | `2cc78c1624d197afcbe0762a61a69e0b53abdc6709001b80d2194e6e7f9b3d24` |
| `eval-v1.py` (runner) | `cbe9db0aa04fc524a5214d147b038c0067ac1ae8ebaca4f3ddf9a2ac4903ea02` |
| `eval-v1-compare.py` (сравнение и правило) | `ed4d7c7bc04a62403363ef976823d26ee3d6db6c92feac124200b51833ba9147` |
| `eval-v1/cases_v1.py` (60 кейсов, чекеры) | `a53da31641e5ecbbda9c8db22d27491e93917d28c693e0b565bfb1634df54b1f` |
| `eval-v1/sandbox_v1.py` (песочница для кода) | `7fc3040059f5aed9951e2c1b77a04e4923a953ac2cc8d4a6dac7d913abcf26f9` |
| `test-eval-v1-cpu.py` (CPU-тесты, 12/12; результат в `eval-v1/cpu-tests.json`) | `df213734bad2ed799310b8750f7a3ab00b8384646eb1350c0fc34e2817709a01` |
| `bf128-fp16compat.json` (= bf128 + `QWEN36_NVFP4_GLOBAL_FP16_COMPAT=1`) | `d8b6a0733d21b5bd43e4bde48e4d953bf2c84e7a6882ab53a3a031ad01d16941` |

Хэши набора и правила:
- suite_sha256 `2e75277e41489dd17ef982e15fc2eba174797b9a91d2beb687dc6d8fc10e2fcf`;
- decision_rule_sha256 `e8f7823f3c1c2dac68f85c56de22fcf4c42174dfde874f38576a3a8329f6dbfb`.

**Набор.** 5 категорий по 12 кейсов, выполняются в этом порядке:
- **code:** 12 функций Python с юнит-тестами;
- **ru:** 12 русских вопросов с однозначным ответом;
- **tool:** 12 сценариев tool_choice=auto. 2–3 функции на выбор, цепочки из 2 вызовов, параллельный вызов, кейс «вызов не нужен», типы integer/number/enum/array. Проверяются id/name, строгий JSON, типизированные аргументы, результат в ответе;
- **agent:** 12 многоходовых историй. Исходный и исправленный запрос, оба должны пройти. Есть правка последнего и раннего сообщения, исправление внутри диалога, история tool-вызовов с новым вызовом, правка кода;
- **long:** 12 синтетических журналов, по 4 на длину ~16K, ~64K и ~120K токенов: 4 факта, подсчёт и сумма 4 отметок, multi-hop с дистракторами, последнее значение из 3 изменений. Иголки стоят на позициях от 1% до 99%.

**Параметры прогона:**
- greedy: temperature 0, top_p 1, seed 4170, thinking off; max_tokens от 128 до 512;
- входы детерминированы, токенизатор во время прогона не нужен;
- в freeze записаны sha256 каждого кейса и всего набора;
- в отчётах только PASS/FAIL, коды отказа, токены и sha256 ответов. Текста ответов нет.

**Песочница для кода (не слабее quality.py).** Код модели исполняется только в отдельном процессе. Защиты:
- AST-политика: без import/class/with/global/декораторов, без имён `__*` и опасных builtins, атрибуты только из белого списка;
- урезанные builtins;
- запуск `python -I -S -B` в пустом временном каталоге, с минимальным env и новой сессией;
- лимиты RLIMIT_CPU 5 с, AS 1 ГиБ, FSIZE 0, NOFILE 32, NPROC 0; wall-таймаут 10 с;
- сеть отключена через `unshare -rn`. В WSL работает, это проверено тестом: connect падает;
- вердикт с nonce, наружу выходят только bool.

## 1. Флаг QWEN36_NVFP4_GLOBAL_FP16_COMPAT в FreeToken

Правка минимальная, 18 строк, файл `work/agents/eval-suite/fp16compat.diff`. Копии до правки лежат в
`work/agents/eval-suite/before/`. Затронуто:
- `models/nvfp4_banks.py`: новая `nvfp4_global_fp32(t)`. При `=1` возвращает `fp32(fp16(g))`, без флага или при `0` — fp32 как раньше. Любое другое значение → `ValueError`, то есть fail-closed;
- её вызывают `_ingest_global` (банки экспертов), `moe/base.global_rows` (pack и все MoE-кернелы, включая Marlin/b12x) и `qwen3_5_moe/weight._DenseReader._check` (shared expert и dense NVFP4).

Хэши после правки, они же `runtime_expectations` во freeze:

| Файл | sha256 |
|---|---|
| nvfp4_banks | `abea022a…f899e6` |
| qwen3_5_moe/weight | `07a08d78…ed79e9` |
| moe/base | `bd01631d…61bd8df` |

**CPU-доказательство:** `work/agents/eval-suite/compat_cpu.json`, `all_ok=true`, около 4 минут через `compat.sh`. Проверка шла
через загрузчики FreeToken по всем 40×256 экспертам и 81 dense `weight_global`.
- **old:** модули из `work/agents/fp32-global/before/` подменены через import hook. Получены fp16-банки.
- **compat:** рабочее дерево с флагом=1. Получены fp32-банки, побитово равные `old.float()`, для 10.49M gate_up и 20.97M down элементов и всех 81 dense.
- **fp32:** без флага. Отличается от compat в 9.99M и 20.03M элементах, то есть флаг реально действует.
- Коды и скейлы во всех трёх режимах побитово одинаковы.

**Почему это даёт побитово старые числа на GPU.** Старые ядра читали fp16 global и сразу точно расширяли его до fp32. Дальше
вся арифметика шла в fp32. Теперь ядро получает то же fp32-значение напрямую. Конфиги ядер фиксированы и не автотюнятся по dtype.
Разница в IR — только отсутствующий `cvt.f32.f16` на входе умножения, на FMA-контракцию он не влияет.

| Ядро | Как global входит в арифметику |
|---|---|
| Triton fused MoE prefill `_prefill_nvfp4_moe_kernel` | `tl.load(g).to(fp32)`, затем `acc*g`, `*w`, bf16 |
| Triton decode `_decode_nvfp4_marlin_kernel` (M=1, production), `_decode_nvfp4_moe_kernel` | то же: `sum*g` |
| Triton NVFP4 linear, shared expert: gemv, splitk, gemm, `_dequant_rows` | `load(g).to(fp32)`, затем `*g`, `*128.0` или `16384.0*g` — точные степени двойки |
| `dequant_nvfp4` | `load(g).to(fp32)`, затем `fp4*scale*g` |
| CUDA bit-exact | v3 делал `__half2float(gl[col])`, v4 читает fp32 `gl[col]`; эпилог тот же `acc*g` |
| `cpu_moe_ext` | `fp16_to_f32` точно обрабатывает субнормали; fp32 `g` даёт то же значение |
| Marlin/b12x pack | `row_global.to(float32)` или `fused_global(...).float()` — те же значения |

**Пропуск env.** `experiment.py` чистит унаследованные `QWEN36_*`, затем ставит `config.env` в `os.environ` до загрузки модели. Флаг доходит.
- `bench.validate_speed_config` его не запрещает.
- `launcher_problems` смотрит стартовый environ MainPID, где флага нет. Значит, `feature_env_in_launcher` не срабатывает.
- `start-experiment.sh` сам копирует `bf128-fp16compat.json` из workspace в `config/`, запуск идёт через `run-experiment.sh`.
- `profile-runtime.py` (production) не трогал. В production флага нет, там fp32 по умолчанию.

## 2. Переменные (WSL, USER)

```bash
W=/path/to/qwen36-workspace; O=$W/outputs/qwen36-api/optimization
N=/home/USER/qwen36-api/optimization; PY=/home/USER/qwen36-api/.venv/bin/python
LOCK=/home/USER/qwen36-api/run/gpu-exclusive.lock
```

## 3. Предусловия (STOP, если не выполнены)

1. R1 (fp32 global) уже прошёл:
   - в `reports/teacher.jsonl` есть строки `teacher-bf128` от fp16-прогона до R1 и от fp32-прогона R1;
   - есть отчёт `compare-logit-reference` (например `logit-reference-r1-vs-pre-r1.json`);
   - есть бэкап `reports/functional-bf128-pre-r1.json`.
2. Units `qwen36-inference` и `qwen36-experiment` неактивны, GPU-lock свободен, другие агенты не меняют `work/FreeToken-opt`.
3. `cd $O && $PY eval-v1.py verify` выдаёт `"ok": true` и `freeze_sha256` = `2cc78c16…`. Если freeze не совпадает, это STOP: правку файлов после freeze не запускать.

## 4. Шаги

### Шаг A. Sync runtime с флагом и CPU-проверки (~5 мин, CPU)

Выполнять в том же порядке, что и Шаг 1/3 harness-v6, с новой меткой:

```bash
bash $O/sync-v5.sh --print-freeze > $O/sync-eval-v1-freeze.sha256        # БЕЗ --skip-runtime
grep -E '  work/FreeToken-opt/python/freetoken/(models/nvfp4_banks.py|models/qwen3_5_moe/weight.py|layers/quantization/moe/base.py)$' $O/sync-eval-v1-freeze.sha256
#   ожидаемо abea022a… / 07a08d78… / bd01631d…
bash $O/sync-v5.sh --dry-run --label eval-v1 --freeze $O/sync-eval-v1-freeze.sha256   # 0 mismatch
bash $O/sync-v5.sh --apply   --label eval-v1 --freeze $O/sync-eval-v1-freeze.sha256   # APPLY PASS
bash $O/sync-v5.sh --verify  --label eval-v1                                           # 0 drift
sha256sum $N/python/freetoken/{models/nvfp4_banks.py,models/qwen3_5_moe/weight.py,layers/quantization/moe/base.py}
cd $O && CUDA_VISIBLE_DEVICES= $PY test-eval-v1-cpu.py | tail -1       # {"passed": 12, "total": 12, ...}
cd $O && $PY eval-v1.py verify                                         # ok true
bash $W/work/agents/eval-suite/compat.sh | tail -1                     # (опц., ~4 мин) "all_ok": true
```

### Шаг B. GPU-доказательство compat на модели (~7 мин GPU, рекомендовано)

```bash
flock -n $LOCK bash -c 'set -e; ! systemctl --user is-active --quiet qwen36-experiment || exit 73
  trap "systemctl --user stop qwen36-experiment" EXIT
  bash '$N'/start-experiment.sh bf128-fp16compat; '$PY' '$N'/check-experiment.py 300
  cd '$N' && '$PY' '$N'/quality.py --name functional-bf128-fp16compat'
$PY - <<'EOF'
import json; R='/home/USER/qwen36-api/optimization/reports/'
a={r['case']:r for r in json.load(open(R+'functional-bf128-pre-r1.json'))['rows']}
b={r['case']:r for r in json.load(open(R+'functional-bf128-fp16compat.json'))['rows']}
for k in a: print(k, a[k]['pass'], b[k]['pass'], (a[k].get('sha256') or a[k].get('content_sha256')) == (b[k].get('sha256') or b[k].get('content_sha256')))
EOF
```

Ожидание: результат 5/5, как до R1. У executable-code, long-facts-16384 и long-facts-120000 хэши совпадают
с pre-R1, то есть генерация побитово прежняя.

Если хэши расходятся, но 5/5, значит после pre-R1 в дереве менялось что-то ещё, кроме global. Флаг это не
опровергает, CPU-доказательство остаётся в силе. Учесть при выводах.

Не использовать имя `functional-bf128`: `quality.py` перезаписывает файл.

### Шаг C. eval-v1 на двух профилях (~15–18 мин GPU каждый, в фоне)

Порядок зафиксирован: сначала bf128, потом bf128-fp16compat. Скрипт сам берёт `flock -n` на LOCK, поэтому обёртку `flock` **не** использовать.

```bash
cd $O && $PY eval-v1.py managed --name eval-v1-bf128 --profile bf128
cd $O && $PY eval-v1.py managed --name eval-v1-bf128-fp16compat --profile bf128-fp16compat
```

Что делает каждый запуск:
- проверяет freeze, исходники, workspace-конфиг и native runtime (3 файла);
- берёт lock, проверяет, что units неактивны;
- выполняет `start-experiment.sh` и `check-experiment.py 600`;
- проверяет launcher и receipt: конфиг профиля и runtime-хэши;
- прогоняет 60 кейсов (~88 запросов);
- останавливает только свой InvocationID;
- пишет `$N/reports/<name>-eval-v1.json` через O_EXCL.

Ожидаемый вывод: `{"status": "VALID", "invalid_codes": [], "summary": {...}}`. Коды выхода:
- 0 — VALID;
- 1 — INVALID;
- 65 — freeze, профиль или runtime не синхронизированы;
- 73 — unit активен;
- 75 — lock занят.

Протокол:
- INVALID можно повторить под новым `--name`;
- VALID-прогон не повторяется и не заменяется. Compare сам проверяет, что взят первый VALID-прогон каждого профиля.

Режим без управления юнитами: `eval-v1.py run --name X --expect-profile P`, как `tool-task-bench --pilot`.

### Шаг D. Сравнение и решение (<1 мин, CPU)

```bash
$PY - <<'EOF'
import json; d=json.load(open('/home/USER/qwen36-api/optimization/reports/logit-reference-r1-vs-pre-r1.json'))['nll']
print('fp16(old)=', d['old_run_id'], ' fp32(new)=', d['new_run_id'])
EOF
cd $O && $PY eval-v1-compare.py --name eval-v1-basis-decision \
  --baseline eval-v1-bf128-fp16compat --candidate eval-v1-bf128 \
  --fp16-teacher-run-id <old_run_id> --fp32-teacher-run-id <new_run_id>
```

Перед запуском убедиться, что old — это fp16-прогон до R1, а new — fp32-прогон R1. Например, по времени в teacher.jsonl.

Результат пишется в `$N/reports/eval-v1-basis-decision.json` через O_EXCL. Поле `decision` принимает одно из значений:
- `fp32`;
- `fp16-compat`;
- `NO_DECISION` — входы несопоставимы, причина в `problems`.

Отчёт также содержит итог и разбивку по категориям, дискордантные пары по id, точный двусторонний p Мак-Немара (общий и по категориям) и NLL с CI95.

## 5. PRE-REGISTERED правило (в freeze, `decision_rule_sha256 e8f7823f…`)

fp32 становится базисом, только если выполнены **все** условия. Иначе базис — **fp16-compat**.

1. **Teacher NLL** fp32 против fp16 на 20 фиксированных teacher-кейсах. Относительный рост perplexity:
   - точечная оценка ≤ +0.1%;
   - верхняя граница CI95 ≤ +1%. Это существующий NLL-гейт compare-quality.py: paired cluster bootstrap, 20000 ресэмплов, seed 4172026.
2. **eval-v1 в целом и по категориям:**
   - PASS(fp32) ≥ PASS(fp16-compat) − 1;
   - в каждой категории нетто-потеря ≤ 1. Нетто-потеря = #(fp16 PASS, fp32 FAIL) − #(fp16 FAIL, fp32 PASS).
3. **Tool calling:** нет ни одного tool-кейса, который проходит на fp16-compat и падает на fp32.

Если входы несопоставимы, решения нет (NO_DECISION). Несопоставимо:
- любой INVALID-прогон;
- разные freeze, набор, исходники, runtime, пакеты или checkpoint;
- конфиги отличаются не только флагом compat;
- неполные NLL-данные;
- взят не первый VALID-прогон.

Мак-Немар только для информации. Обоснование:
- fp32 — математически точный деквант. На CPU он побитово равен FP64-эталону, а у fp16 ошибка до 1.66e-3 на строку. Поэтому без доказательств вреда выбирается fp32.
- Хрупкие кейсы переворачиваются от любого сдвига, поэтому допускается ±1 в сумме. Систематическая деградация видна как нетто −2 в категории.
- tool calling — критичный для агентов приоритет владельца, потери там недопустимы.
- При 60 парах тест Мак-Немара почти не имеет мощности. Поэтому решение принимается по заранее заданным порогам, а не по p-value.
- NLL проверяется на человеческих текстах, а не на собственных генерациях модели, так что смещения в пользу fp16 нет.

## 6. Длительность

| Шаг | Время |
|---|---|
| A | ~5 мин CPU (+4 мин опционально `compat.sh`) |
| B | ~7 мин GPU |
| C | 2 × ~15–18 мин GPU (оценка ниже) |
| D | <1 мин |

Итого ≈ 40–45 мин GPU.

Оценка шага C на один профиль:
- загрузка 3–5 мин;
- длинные кейсы: префилл 4×16K ≈ 6 с, 4×64K ≈ 22 с, 4×120K ≈ 58 с (TTFT bf128 Triton). Всего ≈ 6 мин;
- остальные ~75 запросов ≈ 3–4 мин;
- стоп ~0.5 мин.

Лимит 45 мин на профиль с запасом.

## 7. NOT_RUN и риски

- **GPU-прогоны (B, C, D) не выполнялись** — их запускает родитель.
- **Детерминизм.** Сравнение предполагает, что greedy-генерация на одном профиле воспроизводима. Шум одного прогона не измерялся. Если родитель хочет его оценить, можно сделать третий прогон bf128 под другим именем. В решении он не участвует: по протоколу считается первый VALID-прогон.
- **Флаг покрывает не все загрузчики.** Он стоит в загрузчиках, которые использует Qwen3.6: банки экспертов, `global_rows`, dense `_check`. Пути `loader.nvfp4_parts_ct` (compressed-tensors), gemma4 и glm4 флаг не затрагивает, этой моделью они не используются.
- **Teacher для compat-профиля недоступен.** Имя профиля не в allowlist `engine/qualification.py`. GPU-доказательство compat идёт через хэши functional (шаг B).
- **Сеть в песочнице.** `unshare -rn` доступен в текущем WSL. Если где-то его нет, runner пишет `network_isolated:false`, но остаются AST-политика и урезанные builtins без import.
