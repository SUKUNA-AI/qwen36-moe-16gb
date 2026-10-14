"""Short-route TTFT driver (run by gpu-session.sh under the GPU lock, <= 19 min per session).

For each arm (shortroute-r / -d / -p, see make_configs.py) in the given order: start a FRESH
experiment unit, wait ready, one short warm request + QWEN36_BENCH_WARMUP_S=60 sustained warm-up
(bench.sustained_warmup), then repeats of the SAME greedy workload (identical prompts in every arm):
  fresh prompts of ~64 / 512 / 1024 / 2000 (1 chunk) / 2200 (2 chunks) / 4096 / 8192 total tokens,
  a prefix-hit continuation of the 4096 prompt (bench history_continuation),
  a 16K primer + last-message edit (bench edited_fixtures, ~16.3K cached).
64 forced output tokens each (ignore_eos, temperature 0). Per request: client TTFT, per-delta
arrival times (early ITL), usage, and the joined engine row (prefill/decode GPU ms, prefill H2D/D2D
bytes, decode expert misses, layer-major prefill count). Stops the unit after each arm.

Usage: driver.py SESSION ARMS   e.g. driver.py s1 rdp
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NATIVE = pathlib.Path('/home/USER/qwen36-api/optimization')
HERE = pathlib.Path('/path/to/qwen36-workspace/work/next-update/short-route')
sys.path.insert(0, str(NATIVE))
SESSION_S = 18.5 * 60
MAX_REPEATS = 5
OUT_TOKENS = 64
TARGETS = (64, 512, 1024, 2000, 2200, 4096, 8192)
QUESTION = '\nОбъясни подробно алгоритм поиска аномалий в этих записях. Напиши код на Python и разбор сложности.'


def sh(args, timeout):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, timeout=timeout, text=True, capture_output=True).stdout


def unit_active():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'qwen36-experiment']).returncode == 0


def stream(messages, maximum=OUT_TOKENS):
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
    completion = usage.get('completion_tokens', 0)
    result = {'status': 200, 'usage': usage, 'request_id': rid,
              'ttft_s': times[0] - start if times else None, 'elapsed_s': time.perf_counter() - start,
              'deltas': len(times), 'delta_times_s': [round(t - times[0], 5) for t in times],
              'decode_client_tps': (completion - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None,
              'content_sha256': hashlib.sha256(''.join(text).encode()).hexdigest()}
    attach_metrics(result)
    return result


def slim(case, rep, arm, messages_sha, r):
    e = r.get('engine') or {}
    d = e.get('expert_counter_deltas') or {}
    lm = e.get('layer_major_prefill') or {}
    u = r.get('usage') or {}
    return {'arm': arm, 'case': case, 'rep': rep, 'messages_sha256': messages_sha, 'status': r.get('status'),
            'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens'),
            'cached': r.get('cached_prompt_tokens'), 'ttft_s': r.get('ttft_s'), 'elapsed_s': r.get('elapsed_s'),
            'decode_client_tps': r.get('decode_client_tps'), 'deltas': r.get('deltas'),
            'delta_times_s': r.get('delta_times_s'), 'content_sha256': r.get('content_sha256'),
            'ttft_generated_s': r.get('ttft_generated_s'), 'decode_ack_tps': r.get('decode_ack_tps'),
            'prefill_gpu_ms': e.get('prefill_gpu_ms'), 'prefill_forward_tokens': e.get('prefill_forward_tokens'),
            'decode_gpu_ms': e.get('decode_gpu_ms'), 'decode_forwards': e.get('decode_forwards'),
            'engine_span_s': e.get('engine_span_s'),
            'prefill_h2d_bytes': d.get('prefill_h2d_bytes'), 'prefill_d2d_bytes': d.get('prefill_d2d_bytes'),
            'decode_missing': d.get('decode_missing'), 'decode_active': d.get('decode_active'),
            'lm_prefills_total': lm.get('prefills'), 'lm_last_chunks': lm.get('last_chunks'),
            'lm_last_borrowed_slots': lm.get('last_borrowed_slots'),
            'valid_gpu_expert_slots': e.get('valid_gpu_expert_slots'),
            'engine_metrics_status': r.get('engine_metrics_status')}


def workload(tok, rep):
    """[(case, messages)] for one repeat; identical in every arm. 'follow' entries depend on the previous one."""
    from bench import document, edited_fixtures
    items = []
    for target in TARGETS:
        sysmsg = {'role': 'system', 'content': f'Short-route fixture {rep} {target}. Ответ по-русски.'}
        empty = [sysmsg, {'role': 'user', 'content': QUESTION}]
        overhead = len(tok.apply_chat_template(empty, add_generation_prompt=True, tokenize=True,
                                               enable_thinking=False))
        body = document(tok, max(8, target - overhead), 271000 + 97 * rep + target)
        messages = [sysmsg, {'role': 'user', 'content': body + QUESTION}]
        items.append((f'fresh-{target}', messages))
        if target == 4096:
            follow = messages + [
                {'role': 'assistant', 'content': 'Проверяем типы полей, пропуски и допустимый диапазон задержки. '
                 'Для числовых полей можно использовать медиану и абсолютные отклонения.'},
                {'role': 'user', 'content': 'Добавь обработку отсутствующих значений.'}]
            items.append(('prefixhit-4096', follow))
    body16 = document(tok, 16384, 314159 + 7 * rep)
    case, old, new, _, _ = next(edited_fixtures(body16, 1000 + rep))
    items.append(('edit16k-primer', old))
    items.append(('edit16k', new))
    return items


def run_arm(arm, tok, session, arm_deadline):
    from bench import sustained_warmup
    profile = f'shortroute-{arm}'
    out = HERE / 'results' / session
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f'{arm}.jsonl'
    if unit_active():
        raise SystemExit('qwen36-experiment already active under our GPU lock; refusing')
    since = time.strftime('%Y-%m-%d %H:%M:%S')
    meta = {'arm': arm, 'profile': profile, 'session': session, 'start': time.time()}
    try:
        sh(['bash', NATIVE / 'start-experiment.sh', profile], 60)
        sh([sys.executable, '-B', NATIVE / 'check-experiment.py', '330'], 340)
        meta['ready'] = time.time()
        receipt = json.loads((NATIVE / 'active-run.json').read_text())
        meta['run_id'] = receipt.get('run_id')
        meta['source_sha256'] = receipt.get('source_sha256')
        meta['flags'] = receipt.get('configuration', {}).get('flags')
        meta['env_switches'] = {k: v for k, v in receipt.get('configuration', {}).get('env', {}).items()
                                if 'LAYER_MAJOR' in k or 'SHORTROUTE' in k}
        warm = stream([{'role': 'user', 'content': 'Объясни принцип двоичного поиска.'}], 32)
        meta['warm_status'] = warm.get('status')
        meta['sustained_warmup'] = sustained_warmup(tok, 60)
        meta['measure_start'] = time.time()
        with rows_path.open('a') as f:
            for rep in range(MAX_REPEATS):
                if time.time() > arm_deadline - 30:
                    meta['stopped_at_rep'] = rep
                    break
                for case, messages in workload(tok, rep):
                    sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    r = stream(messages)
                    row = slim(case, rep, arm, sha, r)
                    f.write(json.dumps(row) + '\n')
                    f.flush()
                    print(json.dumps({k: row[k] for k in ('arm', 'case', 'rep', 'prompt_tokens', 'cached', 'ttft_s',
                                                          'decode_client_tps', 'prefill_gpu_ms', 'prefill_h2d_bytes',
                                                          'prefill_d2d_bytes', 'lm_prefills_total')}), flush=True)
                    if r.get('status') != 200:
                        raise RuntimeError(f'request failed: {case} {r}')
            else:
                meta['stopped_at_rep'] = MAX_REPEATS
    except Exception as exc:  # noqa: BLE001
        meta['error'] = repr(exc)
        print('ARM ERROR', arm, repr(exc), flush=True)
    finally:
        meta['end'] = time.time()
        try:
            j = subprocess.run(['journalctl', '--user', '-u', 'qwen36-experiment', '--since', since, '--no-pager', '-o', 'cat'],
                               capture_output=True, text=True, timeout=60).stdout
            keep = [ln for ln in j.splitlines() if any(s in ln for s in (
                'layer-major', 'hit-D2D', 'hit-d2d', 'shortroute', 'Traceback', 'Error', 'error', 'moe_cache', 'prefill_overlap'))]
            (out / f'{arm}.journal.txt').write_text('\n'.join(keep[-400:]) + '\n')
        except Exception as exc:  # noqa: BLE001
            meta['journal_error'] = repr(exc)
        if unit_active():
            subprocess.run(['systemctl', '--user', 'stop', 'qwen36-experiment'], timeout=90)
        (out / f'{arm}.meta.json').write_text(json.dumps(meta, indent=1))


def main():
    session, arms = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/path/to/LLM/qwen36-api/model', local_files_only=True)
    t0 = time.time()
    for i, a in enumerate(arms):
        left = SESSION_S - (time.time() - t0)
        share = left / (len(arms) - i)
        if share < 300:
            print(f'skip arm {a}: only {left:.0f}s left', flush=True)
            continue
        run_arm(a, tok, session, time.time() + share - 20)
    print('driver done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
