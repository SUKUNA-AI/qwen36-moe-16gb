# CCCL на sm_120: что проверено и план большого прогона (21.10.2026)

Всё, что ниже, сделано на RTX 5070 Ti (sm_120) в WSL archlinux, nvcc 13.4 (V13.4.59), CCCL `main` `d416a1153`.
Ничего из этого пока не отправлено в NVIDIA/cccl.

## 1. Аудит `cuda::ptx`: какие инструкции идут через вызов драйвера

Метод: собственные тесты CCCL `libcudacxx/test/libcudacxx/cuda/ptx/ptx.*.compile.pass.cpp` (71 файл) берут адрес
каждой перегрузки каждой обёртки, поэтому компилятор обязан собрать их все. Каждый файл собирается `nvcc -arch=sm_XX
-cubin`, затем в ELF ищутся `__cuda_syscall*`, а в SASS — `CALL.ABS`. Скрипт:
[`tools/cccl-sm120/ptx-syscall-scan.sh`](../../tools/cccl-sm120/ptx-syscall-scan.sh); сырые результаты:
[`results/cccl-sm120-audit/`](../../results/cccl-sm120-audit/). Вся компиляция под одну архитектуру занимает меньше минуты.

| Архитектура | Через вызов драйвера |
|---|---|
| sm_120, sm_120a | `cp.async.bulk` (`.shared::cluster`, unicast) и `cp.async.bulk.tensor` 1D–5D tile (`.shared::cluster`, unicast): 6 вариантов |
| sm_90a, sm_100a | ничего |

Вывод: наш PR [NVIDIA/cccl#12081](https://github.com/NVIDIA/cccl/pull/12081) закрывает ровно все затронутые места
(страницы `cp_async_bulk` и `cp_async_bulk_tensor`). Итоги аудита можно добавить в PR комментарием (≈10 минут).

## 2. Попутная находка: тест `ptx.tcgen05.mma.sp` не собирается под sm_100a на CUDA 13.4

`ptxas error: Function '_Z19test_tcgen05_mma_spPPv' uses single CTA(.cta_group::1) and CTA pair granularity(.cta_group::2) and that is not allowed.`

Все остальные тесты `tcgen05` уже разделены на два файла (`*.cta_group_1` / `*.cta_group_2`), а
`generated/tcgen05_mma_sp.h` (232 + 232 перегрузки) остался одним ядром. Исправление механическое — разделить так же.
Готовых issue/PR не нашлось (поиск «tcgen05 mma sp», «cta_group»). Оговорка: заголовки `generated/` делает
внутренний генератор NVIDIA, они могут захотеть перегенерировать сами. Оценка: ≈1 час на issue + PR.

## 3. NVIDIA/cccl#8534 (`block.sync()` на sm_120, CUDA 13.2) — отложено

- Это баг компилятора из CUDA Toolkit, а не CCCL (`cooperative_groups` в CCCL не входит).
- На nvcc 13.4 два ядра по образцу cudf (`tools/cccl-sm120/block-sync-pilot.cu`, в том числе с невстроенными
  вызовами) дают для `block.sync()` и `__syncthreads()` одинаковый SASS (`BAR.SYNC.DEFER`), без `WARPSYNC.COLLECTIVE`.
- Для воспроизведения нужен nvcc 13.2 и, вероятно, настоящий код cudf: 4–8 часов, шанс маленького repro ≈40%.

## 4. Ресурсы для большого прогона

| | |
|---|---|
| CPU | i7-14700KF, 28 потоков |
| Память WSL | 55 ГиБ (`.wslconfig`: memory ≈ 56.6 ГБ, swap 16 ГБ) |
| Диск | ≈757 ГБ свободно |
| Сборка | cmake 4.4, ninja 1.13, gcc 16 (nvcc 13.4 допускает до gcc 16) |
| Сервер модели | `qwen36-inference` disabled — GPU свободна |
| Не хватает | compute-sanitizer, cuobjdump, nvdisasm, ncu: в `~/qwen36-api/cuda-13.4` только урезанный CUDA для сервера |

Полный CUDA ставится владельцем (нужен пароль sudo), пакет Arch `cuda` 13.4.2, 2.4 ГБ, ставится в `/opt/cuda` и
рабочий CUDA сервера не трогает: `sudo pacman -S cuda`.

## 5. План ABCD (≤10 часов без присмотра)

| Трек | Что | Нагрузка | Оценка |
|---|---|---|---|
| A | Собрать и прогнать тесты CUB, Thrust, libcu++ только под sm_120 | CPU (упор в память: ≈10–12 nvcc параллельно), потом GPU | сборка 3–5 ч (уточнить пилотом на 20 тестах), прогон 1–3 ч |
| B | compute-sanitizer (synccheck, racecheck, memcheck) на отобранной части тестов CUB из A | GPU, в 10–100 раз медленнее обычного | бюджет по времени, 2–4 ч |
| C | Дифференциальный фаззинг алгоритмов CUB против CPU-эталона: scan, scan_by_key (#11167), reduce_by_key, radix sort, segmented sort/reduce, select/partition, histogram; случайные размеры у границ тайлов, сиды в логах | GPU с первой минуты | бюджет по времени до конца окна |
| D | Разбор SASS всех cubin из A: `CALL.ABS`/`__cuda_syscall`, стек и сбросы в локальную память (LDL/STL, `-res-usage`), `WARPSYNC.COLLECTIVE`, графы ветвлений `nvdisasm`; сравнение с sm_90/sm_100 на выборке | в основном CPU, после A | 1–2 ч |

Один скрипт-оркестратор: запуск через `systemd-run --user` (переживает закрытие сессии), лимит памяти ≈48 ГБ,
логи и прогресс на диск, итоговый `summary.md` по-русски. Ничего не отправляется автоматически.

Подготовка со мной: 6–8 часов (установка и пилот 1–1.5 ч, D 1–1.5 ч, C 2–3 ч, B 0.5 ч, оркестратор и
пробный прогон 1 ч). Разбор результатов: 2–3 часа. Если подготовка не успевает целиком — сначала A + B + D
(3–4 часа подготовки), фаззер C — к следующему свободному окну. Подготовку делать в отдельной свежей сессии.

## 6. Обзор открытых задач CCCL под sm_120

(допишется, когда придёт обзор помощника: есть ли sm_120 в CI CCCL, что с #11167, #7611–#7615, #10541, как
принимают внешние PR)
