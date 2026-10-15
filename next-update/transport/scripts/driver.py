"""S4 transport A/B driver: run INSIDE the GPU lock (gpu.sh), one round of arms per call.

For each arm in the given order: start a fresh experiment unit (launch.sh, the S4 overlay), wait
ready, one warm request, a short sustained warm-up, then REPS repeats of the same greedy workload
(identical prompts for every arm of a round, keyed by --round):
  fresh-64, fresh-512 (one chunk, fresh), turn-512 (a chat turn on top of the cached fresh-512:
  short uncached suffix), fresh-1024. 32 forced output tokens each.
Per request: client TTFT, early ITL, decode tok/s and the joined engine row (prefill GPU ms,
H2D/D2D bytes, decode misses). --nsys: one profiled unit, five 4-token requests, then fillers
until the capture window ends and the unit exits (the report is analysed by nsys_h2d.py).

Usage: driver.py --arms t1,c0p --round 0 --tag NAME [--reps 6] [--warmup 20] [--nsys]
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HOME = pathlib.Path.home()
NATIVE = HOME / 'qwen36-api/optimization'
HERE = pathlib.Path(__file__).resolve().parent
OUT = HOME / '.cache/qwen36-cloud-review/s4-results'
NSYS_OUT = HOME / '.cache/qwen36-cloud-review/s4-nsys'
MODEL = os.environ.get('QWEN36_MODEL_DIR', '/path/to/LLM/qwen36-api/model')
sys.path.insert(0, str(NATIVE))
os.environ.setdefault('QWEN36_DATA_ROOT', str(NATIVE))
QUESTION = '\nОбъясни подробно алгоритм поиска аномалий в этих записях. Напиши код на Python и разбор сложности.'
TURN = [{'role': 'assistant', 'content': 'Проверяем типы полей, пропуски и диапазон задержки; '
         'для числовых полей подойдут медиана и абсолютные отклонения.'},
        {'role': 'user', 'content': 'Добавь обработку отсутствующих значений и пример теста.'}]


def unit_active():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'qwen36-experiment']).returncode == 0


def stream(messages, maximum):
    import requests
    from bench import attach_metrics, headers
    payload = {'model': 'qwen36-35b', 'messages': messages, 'max_tokens': maximum, 'seed': 17,
               'temperature': 0, 'top_p': 1, 'chat_template_kwargs': {'enable_thinking': False},
               'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}}
    start = time.perf_counter()
    times, text, usage = [], [], {}
    with requests.post('http://127.0.0.1:1919/v1/chat/completions', json=payload, headers=headers(),
                       stream=True, timeout=(5, 600)) as response:
        if response.status_code != 200:
            return {'status': response.status_code}
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                content = choice.get('delta', {}).get('content')
                if content:
                    times.append(time.perf_counter())
                    text.append(content)
        rid = response.headers.get('x-qwen36-request-id')
    gaps = [b - a for a, b in zip(times, times[1:])]
    result = {'status': 200, 'usage': usage, 'request_id': rid,
              'ttft_s': times[0] - start if times else None, 'elapsed_s': time.perf_counter() - start,
              'itl_early_ms': 1000 * sum(gaps[:8]) / len(gaps[:8]) if gaps else None,
              'itl_late_ms': 1000 * sum(gaps[8:]) / len(gaps[8:]) if len(gaps) > 8 else None,
              'content_sha256': hashlib.sha256(''.join(text).encode()).hexdigest()}
    attach_metrics(result)
    return result


def slim(arm, case, rep, sha, r):
    e = r.get('engine') or {}
    d = e.get('expert_counter_deltas') or {}
    u = r.get('usage') or {}
    return {'arm': arm, 'case': case, 'rep': rep, 'messages_sha256': sha, 'status': r.get('status'),
            'prompt_tokens': u.get('prompt_tokens'), 'cached': r.get('cached_prompt_tokens'),
            'ttft_s': r.get('ttft_s'), 'elapsed_s': r.get('elapsed_s'), 'itl_early_ms': r.get('itl_early_ms'),
            'itl_late_ms': r.get('itl_late_ms'), 'content_sha256': r.get('content_sha256'),
            'prefill_gpu_ms': e.get('prefill_gpu_ms'), 'decode_gpu_ms': e.get('decode_gpu_ms'),
            'decode_forwards': e.get('decode_forwards'),
            'prefill_h2d_bytes': d.get('prefill_h2d_bytes'), 'prefill_d2d_bytes': d.get('prefill_d2d_bytes'),
            'decode_missing': d.get('decode_missing'), 'decode_active': d.get('decode_active'),
            'engine_metrics_status': r.get('engine_metrics_status')}


def fresh(tok, target, seed):
    from bench import document
    sysmsg = {'role': 'system', 'content': f'Transport fixture {seed} {target}. Ответ по-русски.'}
    empty = [sysmsg, {'role': 'user', 'content': QUESTION}]
    overhead = len(tok.apply_chat_template(empty, add_generation_prompt=True, tokenize=True, enable_thinking=False))
    body = document(tok, max(8, target - overhead), 271000 + seed)
    return [sysmsg, {'role': 'user', 'content': body + QUESTION}]


def suffix(tok, base, n, seed):
    """A chat turn on top of the cached ``base`` whose new user message is ~n tokens."""
    from bench import document
    body = document(tok, n, 333000 + seed + n) if n > 4 else 'Продолжай.'
    return base + [{'role': 'assistant', 'content': 'Да.'}, {'role': 'user', 'content': body}]


def workload(tok, rnd, rep):
    seed = 1000 * rnd + 10 * rep
    m512 = fresh(tok, 512, seed + 2)
    items = [('fresh-64', fresh(tok, 64, seed + 1)), ('fresh-512', m512), ('turn-512', m512 + TURN)]
    items += [(f'suffix-{n}', suffix(tok, m512, n, seed)) for n in (8, 16, 32, 64, 128)]
    return items + [('fresh-1024', fresh(tok, 1024, seed + 3))]


def screen(tok, rnd, rep):
    """Long prompts (layer-major from 2049 tokens) and C1 threshold points on a cached prefix."""
    seed = 1000 * rnd + 10 * rep
    m512 = fresh(tok, 512, seed + 2)
    items = [('fresh-512', m512)] + [(f'suffix-{n}', suffix(tok, m512, n, seed)) for n in (200, 350, 480)]
    return items + [('fresh-4096', fresh(tok, 4096, seed + 4)), ('fresh-16384', fresh(tok, 16384, seed + 5))]


def threshold(tok, rnd, rep):
    """C1 crossover: chat turns whose uncached suffix is ~155 / 195 / 235 tokens (~75 tokens of template)."""
    seed = 1000 * rnd + 10 * rep
    m512 = fresh(tok, 512, seed + 2)
    return [('fresh-512', m512)] + [(f'suffix-{n}', suffix(tok, m512, n, seed)) for n in (40, 80, 120, 160)]


def start(arm, mode, out_prefix=None):
    if unit_active():
        raise SystemExit('qwen36-experiment already active under our GPU lock; refusing')
    args = ['bash', str(HERE / 'launch.sh'), arm, mode] + ([str(out_prefix)] if out_prefix else [])
    subprocess.run(args, check=True, timeout=60)
    subprocess.run([sys.executable, '-B', str(NATIVE / 'check-experiment.py'), '400'], check=True, timeout=420,
                   capture_output=True)


def stop():
    if unit_active():
        subprocess.run(['systemctl', '--user', 'stop', 'qwen36-experiment'], timeout=120)


def run_arm(arm, tok, args, out):
    from bench import sustained_warmup
    meta = {'arm': arm, 'round': args.round, 'start': time.time()}
    stem = f'{arm}.r{args.round}.{int(meta["start"])}'
    rows_path = out / f'{stem}.jsonl'
    try:
        start(arm, 'plain')
        meta['ready'] = time.time()
        receipt = json.loads((NATIVE / 'active-run.json').read_text())
        meta['run_id'] = receipt.get('run_id')
        meta['env'] = receipt.get('configuration', {}).get('env')
        meta['warm_status'] = stream([{'role': 'user', 'content': 'Объясни принцип двоичного поиска.'}], 32).get('status')
        meta['sustained_warmup'] = sustained_warmup(tok, args.warmup)
        with rows_path.open('w') as f:
            for rep in range(args.reps):
                for case, messages in {'screen': screen, 'threshold': threshold}.get(args.workload, workload)(tok, args.round, rep):
                    sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    r = stream(messages, args.out_tokens)
                    row = slim(arm, case, rep, sha, r)
                    f.write(json.dumps(row) + '\n')
                    f.flush()
                    print(json.dumps({k: row[k] for k in ('arm', 'case', 'rep', 'prompt_tokens', 'cached', 'ttft_s',
                                                          'prefill_gpu_ms', 'prefill_h2d_bytes', 'prefill_d2d_bytes',
                                                          'itl_early_ms')}), flush=True)
                    if r.get('status') != 200:
                        raise RuntimeError(f'request failed: {case} {r}')
    except Exception as exc:  # noqa: BLE001
        meta['error'] = repr(exc)
        print('ARM ERROR', arm, repr(exc), flush=True)
    finally:
        meta['end'] = time.time()
        journal(meta, out / f'{stem}.journal.txt')
        stop()
        (out / f'{stem}.meta.json').write_text(json.dumps(meta, indent=1))


def journal(meta, path):
    since = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(meta['start']))
    j = subprocess.run(['journalctl', '--user', '-u', 'qwen36-experiment', '--since', since, '--no-pager', '-o', 'cat'],
                       capture_output=True, text=True, timeout=60).stdout
    keep = [ln for ln in j.splitlines() if any(s in ln for s in (
        'hit-D2D', 'Traceback', 'Error', 'error', 'WARNING', 'moe_cache', 'prefill', 'S4', 'expert'))]
    path.write_text('\n'.join(keep[-400:]) + '\n')


def run_nsys(arm, tok, args, out):
    NSYS_OUT.mkdir(parents=True, exist_ok=True)
    prefix = NSYS_OUT / f'{args.tag}-{arm}'
    meta = {'arm': arm, 'nsys': str(prefix), 'start': time.time(), 'requests': []}
    try:
        start(f'{arm}-nsys', 'nsys', prefix)
        seed = 1000 * args.round
        m512 = fresh(tok, 512, seed + 2)
        plan = [('warm', [{'role': 'user', 'content': 'Объясни принцип двоичного поиска.'}]),
                ('fresh-64a', fresh(tok, 64, seed + 1)), ('fresh-64b', fresh(tok, 64, seed + 5)),
                ('fresh-512', m512), ('turn-512', m512 + TURN)]
        for case, messages in plan:
            r = stream(messages, 4)
            meta['requests'].append(slim(arm, case, 0, '', r))
            print(case, r.get('ttft_s'), flush=True)
        for i in range(40):
            if not unit_active():
                break
            try:
                stream(fresh(tok, 64, seed + 100 + i), 64)
            except Exception:  # noqa: BLE001 -- the capture end kills the server mid-request
                break
        deadline = time.time() + 900
        while unit_active() and time.time() < deadline:
            time.sleep(5)
    except Exception as exc:  # noqa: BLE001
        meta['error'] = repr(exc)
        print('NSYS ERROR', arm, repr(exc), flush=True)
    finally:
        meta['end'] = time.time()
        journal(meta, out / f'{arm}.nsys.journal.txt')
        stop()
        (out / f'{arm}.nsys.meta.json').write_text(json.dumps(meta, indent=1))


def run_trace(arm, tok, args, out):
    """D3a: eager decode with the route-trace hook; long decodes on short, 4K and 16K prompts."""
    meta = {'arm': arm, 'start': time.time(), 'requests': []}
    try:
        start(arm, 'plain')
        stream([{'role': 'user', 'content': 'Объясни принцип двоичного поиска.'}], 16)
        for i, (case, target) in enumerate((('short', 64), ('short', 512), ('4k', 4096), ('16k', 16384),
                                            ('short', 128), ('4k', 4096))):
            r = stream(fresh(tok, target, 5000 + 17 * i), args.trace_tokens)
            meta['requests'].append(slim(arm, case, i, '', r))
            print(case, target, r.get('ttft_s'), r.get('itl_late_ms'), flush=True)
        stream([{'role': 'user', 'content': 'Готово?'}], 4)  # its prefill flushes the last trace
    except Exception as exc:  # noqa: BLE001
        meta['error'] = repr(exc)
        print('TRACE ERROR', repr(exc), flush=True)
    finally:
        meta['end'] = time.time()
        journal(meta, out / f'{arm}.trace.journal.txt')
        stop()
        (out / f'{arm}.trace.meta.json').write_text(json.dumps(meta, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', required=True)
    p.add_argument('--round', type=int, default=0)
    p.add_argument('--tag', required=True)
    p.add_argument('--reps', type=int, default=6)
    p.add_argument('--warmup', type=float, default=20)
    p.add_argument('--nsys', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--out-tokens', type=int, default=32)
    p.add_argument('--workload', choices=('short', 'screen', 'threshold'), default='short')
    p.add_argument('--trace-tokens', type=int, default=256)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms.split(','):
        (run_trace if args.trace else run_nsys if args.nsys else run_arm)(arm, tok, args, out)


if __name__ == '__main__':
    main()
