# Финальный Nsight: Systems → выбор цели по critical path → Compute

Подготовлено 2026-10-13 задачей nsight-prep; исправлено в тот же день после независимой проверки (fix1: разрез по сценариям, выбор Compute-экземпляра по форме и запросу, проверка M, защита `run` от устаревшего указателя, предпроверка native-копий). **Ни модель, ни GPU, ни profiler в этой задаче не запускались (NOT_RUN).** Здесь только подготовленные конфиги, launcher, анализатор и порядок действий. Ни одно число ниже не является скоростью модели; скорость принимается только парными прогонами без profiler (`matrix.py`/`compare-ab.py`).

## 0. Коротко о терминах

**«Подгрузка промахнувшихся экспертов по PCIe».** В каждом MoE-слое router выбирает для токена 8 из 256 экспертов. Полный банк весов экспертов NVFP4 (около 16.93 GiB) лежит в pinned RAM. В VRAM помещается только кэш слотов (у main-candidate 2998 слотов, около 4.96 GiB). Если выбранного эксперта нет в слотах, это промах кэша. Тогда его веса нужно доставить из RAM в VRAM по шине PCIe: либо копией H2D, либо ядром `fast_index_copy_multi`, которое читает mapped pinned memory. В профиле это категории `expert_mapped_host_gather` и `memcpy_h2d`, а решение «кого держать в слотах» принимает ядро `expert_lru_metadata`. Шина медленнее видеопамяти примерно на порядок: калибровка gather дала около 24.5 GB/s, а VRAM RTX 5070 Ti — 896 GB/s по атрибутам устройства в trace. В старом baseline под profiler (64K) этот gather занял 40.9% GPU critical chain decode-шагов CUDA graph (`model-baseline-analysis-v2.json`). Это гипотеза о приоритете, а не вывод для main-final: конфигурация, ratio и число слотов другие. Атрибуты устройства из trace: 70 SM, L2 48 MiB, 1 async copy engine под WSL.

**Critical path.** Внутри каждого forward анализатор идёт от последней завершившейся GPU-операции назад к предшественнику, который закончился позже всех среди начавшихся раньше. Каждая операция получает только хвост после конца предшественника. Вклады вместе с внутренними простоями GPU дают ровно длительность forward. Сумма длительностей ядер здесь не используется: перекрытые ядра и копии на других потоках в путь не попадают.

## 1. Предпосылки (без них не запускать)

1. **Выбран main-final.** Его выбирают только по пройденным quality+performance gates матрицы final3-v5. До этого готовы конфиги для трёх вариантов: `profile-final-bf128.json`, `profile-final-main-retain.json`, `profile-final-main-retain2048.json`. Для любого будущего варианта (в том числе CUDA-кандидата) конфиг строится той же командой (CPU, около 3 с):
   `python profile-request.py plan --variant <имя-конфига> --write` → `profile-final-<имя>.json`.
   Параметры варианта копируются без изменений. Добавляются только `env.QWEN36_NSYS_FORWARDS`, блок `nsys` и `profile_plan` с SHA256 исходного конфига. Существующие конфиги не правятся.
2. **Код native равен замороженному кандидату.** Runtime/source должны совпадать с тем, что прошло квалификацию: в `active-run.json` `source_sha256` и библиотеки — как в receipts выбранного варианта. Профиль другого кода — не профиль main-final.
3. **Native-копии обновлены.** `start-experiment.sh` при каждом старте копирует `optimization/*.sh` и `*.json` (profile.sh и конфиги обновляются сами), а `*.py` копирует только `sync.sh`. На 2026-10-13 22:30 native `profile-request.py` и `analyze-model-nsys.py` — старые версии (sha256 7ad9efc0… и 1eac87c1… против workspace). Шаблон `final-profile-example.sh` до старта сверяет sha256 native и workspace для `profile-request.py`, `analyze-model-nsys.py`, `bench.py` и при расхождении отказывает (exit 78). Иначе старый `profile-request.py` упал бы на `run` уже после запуска модели под nsys, запросы не ушли бы, capture не закончился бы, и GPU lock держался бы бесконечно.
4. **`start-experiment.sh` должен направлять `profile-final-*` в profile.sh.** Это чужой файл; нужна одна строка (finding в `nsight-prep-results-v2.json`):
   ```
   -case "${1:-}" in profile64|profile-cuda128|profile-main128) launch=profile.sh;; esac
   +case "${1:-}" in profile64|profile-cuda128|profile-main128|profile-final-*) launch=profile.sh;; esac
   ```
   Без неё `start-experiment.sh` запустит модель через `run-experiment.sh`, то есть **без profiler**. Шаблон отказывает ещё до старта (grep маршрута в native `start-experiment.sh`). Если модель всё же запущена в обход шаблона, `profile-request.py run` не шлёт запросы. `active-run.json` для этого не годится: его пишет `experiment.py` при любом способе запуска, с тем же `profile` и `configuration`. Отказ дают проверки самого unit и указателя:
   - `ExecStart` unit `qwen36-experiment` (`systemctl --user show`) должен заканчиваться на `.../profile.sh <профиль>`;
   - `invocation_id` в указателе `reports/profile-final-current.json` (profile.sh пишет `$INVOCATION_ID` от systemd) должен совпасть с `InvocationID` текущего unit. Указатель, оставшийся от прошлого запуска того же профиля, не проходит;
   - `started_utc` указателя и `active-run.json` не раньше старта unit (`ActiveEnterTimestampMonotonic`, допуск 2 с), и указатель не позже `active-run.json`;
   - `<output_base>.run-receipt.json` ещё не существует.
5. **GPU.** `qwen36-inference` и `qwen36-experiment` inactive. Вся последовательность (старт unit → запросы → ожидание выхода unit → анализ) выполняется под `flock -n /home/USER/qwen36-api/run/gpu-exclusive.lock`. Скорость в это время никто не меряет.

## 2. Файлы

| Файл | Роль |
|---|---|
| `profile.sh` | Launcher внутри unit. Старые fixture (`profile64`, `profile-cuda128`, `profile-main128`) сохранили опции nsys. Новое: `umask 077`, уникальные имена с UTC-штампом (существующие отчёты не перезаписываются: `--force-overwrite=false`), chmod 0600, sidecar `<base>.profile-receipt.json` (версия, аргументы, SHA256 файлов), указатель `reports/profile-final-current.json`. Для `profile-final-*` валидирует блок `nsys` (tool `nsys` или `ncu`) и передаёт аргументы без shell-парсинга. |
| `profile-final-<variant>.json` | Те же параметры варианта + `QWEN36_NSYS_FORWARDS` + `nsys{graph_nvtx:precapture, cuda_event_trace:false}` + план. |
| `profile-request.py` | `plan` (CPU: токенизатор, верхняя граница forward), `run` (нагрузка на уже запущенный unit; сам ничего не запускает и не останавливает; проверки из §1.4), `ncu-config` (Compute-конфиг из приватного списка целей схемы 3: `--scope`, `--rank`, `--reference`, метрики-отпечатки). Режим `--length/--output` работает как раньше. |
| `analyze-model-nsys.py` | Схема 3: forward/запросы, critical path по фазам и отдельно по каждому сценарию (`by_request`), слоям и категориям, двойной Sampler, слои graph-узлов, ncu-кандидаты с точным экземпляром; см. `analyze-model-nsys.md`. |
| `work/agents/nsight-prep/final-profile-example.sh` | Шаблон всей последовательности под GPU lock: предпроверки unit, маршрута, sha256 native = workspace, конфига. |

## 3. Сценарий одного профиля

Окно capture открывает `cudaProfilerStart` в первом forward, поэтому прогрев попадает в trace и отделяется анализатором как запросы `warmup*`. Все документы синтетические (`bench.document`), без prompt cache hit. Sampling: seed 17, temperature 0.6, top_p 0.95, thinking off, `ignore_eos`.

| Запрос | Документ, токенов | Prompt (план) | Output | Назначение |
|---|---:|---:|---:|---|
| warmup_short | 256 | 278 | 32 | первый Sampler, первые decode graphs |
| warmup_chunk | chunk+1536 | 3606 / 5654 | 16 | прогрев полного чанка M=chunk и остатка |
| warm_decode | 256 | 278 | 128 | **прогретый короткий decode** (M=1) |
| fresh_16k | 16384 | 16405 | 64 | **свежий 16K**: prefill-чанки + decode на 16K |
| fresh_120k | 120000 | 120022 | 64 | **свежий 120K**: длинный prefill + decode на 120K |
| filler | 256 | 278 | 512 | поглощает запас; его обрывает конец capture |

`QWEN36_NSYS_FORWARDS` — верхняя граница числа forward до конца fresh_120k: `ceil(prompt/chunk) + output + 3` на запрос. Запас +3 покрывает overlap-шаг, короткий хвостовой чанк prefix-margin/retain и разрез по выравниванию. bf128 и main-retain2048 (chunk 2048): **391**; main-retain (chunk 4096): **358**. Максимальный запас 23 forward меньше минимума filler (512). После `cudaProfilerStop` nsys (`--capture-range-end=stop-shutdown --kill=sigterm`) сам завершает модель и пишет отчёт, поэтому unit не упирается в `TimeoutStopSec=30`.

## 4. Порядок выполнения

```bash
# CPU, workspace: при необходимости конфиг для выбранного варианта
python /path/to/qwen36-workspace/outputs/qwen36-api/optimization/profile-request.py plan --variant <main-final> --write
# после native sync и правки start-experiment.sh; всё под GPU lock:
flock -n /home/USER/qwen36-api/run/gpu-exclusive.lock \
  bash /path/to/qwen36-workspace/work/agents/nsight-prep/final-profile-example.sh profile-final-<main-final>
```

Шаблон проверяет оба unit, маршрут и пути в native start-experiment.sh, равенство sha256 native и workspace для `profile-request.py`, `analyze-model-nsys.py`, `bench.py` и наличие конфига; только потом запускает `start-experiment.sh`. Затем `profile-request.py run` проверяет:

- production не активен;
- готовность `/v1/models`;
- что `active-run.json` описывает именно этот конфиг;
- что unit запущен как `profile.sh <профиль>`, а указатель profile.sh записан этим же запуском unit (InvocationID, время старта), и receipt ещё нет (§1.4);
- что SHA256 синтетических сообщений совпадает с планом.

Если `run` отказал или упал, шаблон пишет причину и продолжает ждать выхода unit, не отпуская lock; остановить unit — явное действие владельца (`systemctl --user stop qwen36-experiment.service`).

После этого он шлёт запросы и ждёт выхода unit. GPU lock не отпускается, пока unit активен. Receipt (`<base>.run-receipt.json`, 0600) содержит только числа и хэши, флаги `measured_requests_completed` и `experiment_unit_inactive_after_run`. Затем на CPU:

```bash
root=/home/USER/qwen36-api/optimization; py=/home/USER/qwen36-api/.venv/bin/python
base=<output_base из reports/profile-final-current.json>
$py -I $root/analyze-model-nsys.py --input $root/reports/$base.sqlite \
  --run-receipt $root/reports/$base.run-receipt.json \
  --output $root/reports/$base.analysis.json --private-targets $root/reports/$base.targets.json
# в workspace копировать только обезличенный $base.analysis.json (новое имя, без перезаписи)
```

Перед чтением результата проверить в analysis: `requests[*].label` совпадают с планом (warmup_short, warmup_chunk, warm_decode, fresh_16k, fresh_120k, filler); `requests[*].capture_complete` равно true для всех измеряемых запросов; `forward_instances.gpu_ops_without_instance` мало; `layer_attribution.method_counts` содержит `graph_node_capture_time_nvtx` для decode graph. Если последнего нет, precapture не сработал: слои decode неизвестны, категории остаются достоверными.

**Если `nvtx-precapture` (экспериментальный режим nsys 2026.3.2) завис, упал или раздул память cgroup** (startup peak уже около 27.7 GiB при MemoryHigh 32G), сделать fallback: `plan --variant <v> --graph-nvtx live --write` → `profile-final-<v>-live`. Decode-слои будут `unattributed`, всё остальное то же.

Опционально в тот же GPU-сеанс: такой же профиль `profile-final-bf128` как эталон состава пути. Сравнивать только доли и состав, не время.

### Почему такие опции nsys

`--trace=cuda,nvtx --sample=none --cpuctxsw=none`: OSRT выключен (известное зависание injected helper на `/usr/bin/file`; `experiment.py` дополнительно снимает LD_PRELOAD у `file`). `--cuda-graph-trace=node:nvtx-precapture`: узлы decode graph видны по отдельности, а NVTX во время capture привязывает узел к `Layer_N`. `--cuda-event-trace=false`: device-side трассировка событий может вносить ложные зависимости между потоками; старый baseline записан с ней (таблица CUDA_EVENT), анализатор её не использует. `--wait=primary`, `--export=sqlite`: SQLite создаётся в том же unit до его завершения.

## 5. Как выбирать следующую CUDA-цель (по critical path, не по сумме ядер)

Входные данные — **по сценариям**: `critical_path.by_request[]` для `warm_decode`, `fresh_16k`, `fresh_120k`, в каждом `phases.prefill` (окно чанков этого запроса) и `phases.decode` (сумма периодов decode-шагов этого запроса, последний шаг исключён) с разрезами `by_category`, `by_layer_kind_category`, `by_layer` и своими `ncu_candidates`. `critical_path.phases` — только справочная смесь всех измеряемых запросов: decode взвешен числом шагов (warm_decode 127 периодов против 63 у fresh_16k/fresh_120k), prefill включает и единственный короткий чанк warm_decode. По смеси цель не выбирать: например, full-attention decode, значимый только на 120K, в ней размывается короткими шагами. У агрегатных `ncu_candidates` смотреть `critical_ms_by_request` и `top_request_label`.

1. **Сценарий × фаза.** Для каждого из шести срезов (warm_decode, fresh_16k, fresh_120k) × (prefill, decode) отдельно: decode-период определяет tokens/s на этом контексте, окно prefill — TTFT документа такой длины (у warm_decode prefill — один короткий чанк, для TTFT не показателен). Код, русский, tool calling, длинные документы и история равноприоритетны, поэтому смотреть все срезы. Цель одного среза не должна ухудшать другие.
2. **Сначала исключить не-ядерные узкие места.**
   - Большой `gpu_idle_between_forwards_ms` или `gpu_idle_inside_forward_ms` означает, что путь упирается в host: scheduler, sampler, sync, launch. Нужен CPU/launch-анализ, а не Compute.
   - Если первое место у `expert_mapped_host_gather` или `memcpy_h2d`, путь упирается в PCIe/кэш экспертов. Это задача transport/prefetch/placement (`cuda-decode-prefetch`, `cuda-prefill-transport`), а не математики ядра. Compute на gather-ядре полезен только чтобы подтвердить, что системная память (PCIe) насыщена.
3. **Порог по Амдалу — внутри сценария.** Критерий приёмки — не меньше 5% полного запроса. Ускорение части с долей `s` в `k` раз даёт выигрыш `s·(1−1/k)`. При реалистичных `k≈1.5–2` нужна доля `s ≥ 10–15%`. Доля берётся из `by_request[<сценарий>]`: `share_of_request_critical_percent` (от окна prefill + периодов decode этого запроса) для критерия «полный запрос» и `share_of_phase_critical_percent` для tokens/s или TTFT. Цель, проходящая порог хотя бы в одном сценарии (например, только в fresh_120k), допустима, если в остальных не ухудшает. Цель с меньшей долей не брать, даже если ядро «тяжёлое» по сумме длительностей.
4. **Брать категорию + форму + сценарий.** Каждая строка кандидата — одно ядро, конфигурация запуска и класс формы (`decode_M1`; `prefill_full_chunk` = не последний чанк, ожидаемо M = chunk 2048/4096; `prefill_last_chunk` = остаток). Если у одного семейства значимы несколько форм или сценариев, профилировать каждую нужную пару отдельно (`ncu-config --scope <сценарий>`).
5. **Не путать перекрытия.** Ядро, скрытое за другой работой (на другом потоке или под PDL), в путь не входит, и его ускорение wall time не изменит. `concurrency_wall_share` — второй взгляд: доли делятся поровну в перекрытиях и в сумме дают объединение активности. `forward_overlap_with_next_ms` показывает, насколько следующий forward уже идёт параллельно.
6. **Сверка с уже отвергнутым.** CUDA MoE (качество), B12x, FP8 KV, complement, fused QK/RoPE/KV (нет выигрыша модели), MTP — без новой гипотезы не повторять. Dense FP8 projections, GDN layout copies, MoE decode epilogue, sampler — кандидаты из `remaining-layers-review.md`, но порядок задаёт только новый путь.

## 6. Nsight Compute на процессе модели

1. **Цель.** Из `<base>.targets.json` (приватно, схема 3) взять строку: `--scope all_measured` (ранги агрегата) или `--scope <сценарий>` (ранги внутри warm_decode/fresh_16k/fresh_120k, предпочтительно по §5). Затем `profile-request.py ncu-config --variant <v> --targets <base>.targets.json --scope <s> --rank k [--nvtx-phase] [--filter-mode per-launch-config] [--clock-control none] --write`. Команда создаёт `config/profile-final-<v>-ncu<k>[-<сценарий>].json` только в native (0600): там есть symbol ядра. Экземпляр — запуск с той же фазой, классом формы и полной конфигурацией запуска на critical chain запроса `instance_request_label` (для `all_measured` — запрос с наибольшим critical-временем ключа), с длительностью ближе к медиане. План усекается до этого запроса включительно (плюс filler), поэтому порядковые номера запусков те же, что в Systems. Если индекс запроса в trace не совпадает с меткой плана, или файл целей схемы 2, команда отказывает. Добавляется `QWEN36_STALL_SECONDS=1800`, иначе engine-stall watchdog (300 с) может принять долгую replay за зависание.
2. **Запуск.** Тот же шаблон: `flock ... final-profile-example.sh profile-final-<v>-ncu<k>`. profile.sh собирает:
   ```
   ncu --target-processes all --target-processes-filter exclude:file --profile-from-start off --kill 1
       --replay-mode kernel --graph-profiling node --kernel-name-base demangled --kernel-name regex:^<имя>$
       [--nvtx --nvtx-include 'Qwen36\/decode/' | 'Qwen36\/prefill/']
       --filter-mode global|per-launch-config --launch-skip <N> --launch-count 1
       --cache-control all --clock-control base --import-source no --section-folder <root>/sections
       --section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart
       --section MemoryWorkloadAnalysis_Tables --section ComputeWorkloadAnalysis --section Occupancy
       --section WarpStateStats --section SourceCounters --section SchedulerStats --section LaunchStats
       --metrics gpu__time_duration.sum,smsp__inst_executed.sum,smsp__inst_executed_pipe_tensor.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum
   ```
   Имена метрик проверены офлайн: `ncu --query-metrics --chips gb203`. `--no-metrics` убирает список, если ncu на живом процессе его отвергнет.
   `--profile-from-start off` ограничивает профилирование окном `cudaProfilerStart/Stop`, которое engine открывает по `QWEN36_NSYS_FORWARDS`. `--kill 1` завершает модель после нужного числа результатов. Потом `ncu --import` создаёт приватные `details.csv` и `raw.csv`.
3. **Номер запуска.**
   - Глобальный фильтр без NVTX: `launch_skip_same_name_in_window`.
   - С `--nvtx-phase`: `launch_skip_same_name_same_phase`. В имени `Qwen36/decode` есть `/`, поэтому в синтаксисе ncu push/pop он экранирован, а завершающий `/` обязателен. Как ncu применяет push/pop NVTX к узлам CUDA graph, документация не описывает: проверить на первом запуске.
   - С `per-launch-config`: `launch_skip_same_launch_config`. Отбор и подсчёт идут отдельно по каждой паре grid/block/shared memory. Так удобно взять одно M=1 и одно M=chunk-ядро одного семейства.
4. **Проверка, что профилирован нужный экземпляр (M), а не только нужная конфигурация.**
   - Необходимое условие: grid/block/shared memory в LaunchStats равны строке цели. Не совпали — неверный skip или фильтр.
   - Этого недостаточно. Persistent-ядра (grid = число SM) и decode при разном контексте имеют одну конфигурацию для разных M и контекстов. Поэтому смотреть `launch_config_identifies_instance_group` и `launch_config_groups` в строке цели.
   - Отпечаток длительности: `r = gpu__time_duration / instance_duration_us`. Ожидается `0.67 ≤ r ≤ 2` (под ncu базовые частоты и холодный кэш). Кроме того, `gpu__time_duration` должна быть ближе (в логарифме) к p50 выбранной группы, чем к p50 любой другой группы той же конфигурации. Иначе экземпляр неверный.
   - Если другая группа отличается по p50 меньше чем в 1.5 раза, длительность не различает экземпляры. Тогда снять опорный запуск: `ncu-config ... --rank k --reference <N>` (N — номер группы в `launch_config_groups`) и сравнить `smsp__inst_executed.sum` (не зависит от частот и кэша). Для GEMM-подобных ядер отношение ≈ отношению M (например, 2048 против 278 → ≈7). Для attention decode отношение растёт с контекстом. Если отношение ≈1, формы для этого ядра по работе неразличимы, и выбор между ними на вывод не влияет.
   - Кандидатов ранжирует Systems, а не Compute.
5. **Формы M.** Decode: batch 1 в CUDA graph, M=1. Prefill: M=chunk (2048 у bf128/main-retain2048, 4096 у main-retain) и последний чанк-остаток. Для MoE grouped GEMM M на эксперта — число назначенных токенов (в среднем около chunk·8/256), а не chunk. Размеры из handoff: hidden 2048; full attention q+gate/k/v 9216; GDN QKVZ 12288; out_proj 4096→2048; shared gate/up 2048→1024, down 512→2048.
6. **Что читать.**
   - SpeedOfLight: SM % против memory % — compute- или memory-bound.
   - MemoryWorkloadAnalysis + Tables: байты DRAM/L2/L1, hit rate, system memory (PCIe) у mapped gather.
   - ComputeWorkloadAnalysis: загрузка tensor (FP8/FP4 MMA) против FMA/ALU.
   - Occupancy: теоретическая и достигнутая, что ограничивает (регистры, smem, блоки).
   - LaunchStats: волны на 70 SM. Для M=1 ядро с числом CTA меньше 70 недогружает GPU, кандидат на split-K.
   - WarpStateStats: причины stall (long scoreboard, barrier, math throttle, MIO).
   - SchedulerStats: eligible warps и использование issue slots.
   - SourceCounters: горячие SASS-инструкции, uncoalesced доступ, bank conflicts.

   Ориентиры устройства: 896 GB/s, L2 48 MiB, 70 SM, 48 warps/SM, 100 KiB smem/SM.
7. **Ограничения Compute.**
   - Kernel replay сериализует ядро, сбрасывает кэши (`--cache-control all`, «холодный» L2) и фиксирует частоты (`--clock-control base`). `gpu__time_duration` из ncu не равен времени в модели, сравнивать только относительные метрики.
   - Если блокировка частот под WSL недоступна, использовать `--clock-control none` и отметить это в отчёте.
   - Kernel replay сохраняет и восстанавливает память, которую пишет ядро; при свободных около 1.3–1.5 GiB VRAM возможен отказ. Тогда допускается отдельный ncu-конфиг с ratio на 0.03 ниже, с явной пометкой: это меняет число слотов экспертов, поэтому для gather/MoE-путей результаты несопоставимы.
   - Длительности ядер под ncu не складывать, в wall time не переводить. Kernel microbenchmark не равен скорости модели.

## 7. Чего не делать

- Не мерить скорость под Nsight.
- Не суммировать перекрытые длительности.
- Не выбирать цель по `share_sum_kernel_percent`.
- Не профилировать без lock и параллельно с другими GPU-задачами.
- Не публиковать raw `.nsys-rep`, `.sqlite`, `.ncu-rep`, `targets.json`, `details.csv`/`raw.csv` и ncu-конфиги с symbol.
- Не перезаписывать старые отчёты: имена уникальны, `--force-overwrite=false`.
- Не запускать production и experiment одновременно.

## 8. Оценка времени (оценка, не измерение)

Старт модели без profiler около 87–112 с, под nsys дольше. Prefill около 130K без profiler — около 51 с (boundary TTFT main-candidate), под nsys медленнее. Около 300 decode-шагов. Экспорт SQLite на примерно 0.6 млн GPU-записей занимает минуты. Анализатор на синтетической записи такого размера (563 тыс. ядер) вместе с приватными целями: около 15 с CPU и 1.5 GiB RSS. Итого порядка 10–15 мин на один Systems-профиль и около 5–10 мин на каждую Compute-цель.

## 9. Что проверено в этой задаче (CPU)

- `test-analyze-model-nsys-cpu.py`: 29/29 (10 схемы 1 + 16 схемы 2 + 3 схемы 3: полный чанк не путается с коротким при одинаковом grid, разрез по сценариям в сумме равен агрегату, decode-цель указывает на шаг 120K). Те же новые тесты на анализаторе до fix1 падают (3 error + 1 fail).
- Dry-test `profile.sh` с поддельными nsys/ncu: 33/33 (+ `invocation_id` в указателе, `--metrics`, отказ на плохую метрику).
- `profile-request.py` (`work/agents/nsight-prep/fix1/scripts/test_profile_request_v2.py`): 15/15 — проверки `run` (unit без profile.sh, устаревший указатель того же профиля, указатель без invocation id, порядок времён, существующий receipt), ncu-config (экземпляр fresh_16k вместо warm_decode, scope, reference, отказ на схему 2 и на несовпадение порядка запросов), детерминизм трёх `profile-final-*.json`.
- Dry-test шаблона с поддельными systemctl/python/start-experiment: 12/12 (старый native `profile-request.py`, отсутствующий анализатор, изменённый `bench.py` → exit 78 до старта).
- Схема 1 на `model-baseline.sqlite` побайтно равна прежнему выводу; схема 3 равна схеме 2 во всех разделах, кроме добавленных.
- Стресс анализатора на синтетической записи финального размера (562840 ядер) вместе с приватными целями: 15.4 с CPU, около 1.5 GiB RSS.

Live Systems/Compute, `nvtx-precapture` на реальной модели, синтаксис ncu NVTX на graph-узлах, `--kill 1`, `--target-processes-filter exclude:file` (только исключающий фильтр) и `stop-shutdown` под systemd — **НЕ ПРОВЕРЕНО**. Если ncu не профилирует python из-за фильтра, убрать его: `experiment.py` и так снимает LD_PRELOAD у `/usr/bin/file`.
