"""Two-phase owner-local prefix quality gate; never starts/stops a model.

Fresh and cached phases MUST use separate clean runs of the same profile/source.
Synthetic requests exist only in memory. Reports contain numeric verdicts/hashes,
never prompts, responses, tool arguments, auth headers, or server error bodies.
"""
import os
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

VERSION = 1
MAXIMUM = 512
SEED = 74017
TOOL = [{'type': 'function', 'function': {
    'name': 'lookup_fixture', 'description': 'Return a safe synthetic project record. No external actions.',
    'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}},
                   'required': ['key'], 'additionalProperties': False}}}]
GOLD = {'code': 'АЛЬФА-917', 'count': 23, 'owner': 'Ирина', 'price': 14, 'total': 322}
EDITED = dict(GOLD, count=29, price=17, total=493)
QUESTION = ('Верни только один JSON-объект без Markdown и пояснений, с полями code, count, owner, '
            'price, total. Числа должны быть целыми. total=count*price. '
            'Используй актуальные факты проекта, а не отвлекающие записи. Больше tools не вызывай.')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def duplicate_safe(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def strict_object(text):
    def invalid_constant(_value):
        raise ValueError('nonfinite')
    if not isinstance(text, str) or len(text) > 16384:
        raise ValueError('invalid_text')
    obj = json.loads(text, object_pairs_hook=duplicate_safe, parse_constant=invalid_constant)
    if not isinstance(obj, dict):
        raise ValueError('not_object')
    return obj


def verdict(text, expected):
    try:
        obj = strict_object(text)
        fields = {key: key in obj and type(obj[key]) is type(value) and obj[key] == value
                  for key, value in expected.items()}
        keys = set(obj) == set(expected)
        return {'strict_json': True, 'keys_equal': keys, 'field_equal': fields,
                'pass': keys and all(fields.values()), 'parsed_sha256': digest(obj)}
    except (ValueError, TypeError, RecursionError):
        return {'strict_json': False, 'keys_equal': False,
                'field_equal': {key: False for key in expected}, 'pass': False}


def common_prefix(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


def tool_anchor_boundary(target_ids, primer_ids, opener_ids):
    if len(opener_ids) != 1:
        return None
    # The tools instruction itself contains an example <tool_call>. It is
    # shared input, never the generated decode anchor we intend to verify.
    start = common_prefix(target_ids, primer_ids)
    return next((i+1 for i in range(start, len(target_ids)) if target_ids[i] == opener_ids[0]), None)


def render_ids(tokenizer, messages, tools=None):
    kwargs = {'enable_thinking': False}
    if tools is not None:
        kwargs['tools'] = tools
    # Exact server rendering: apply_chat_template tokenize=False, then encode
    # without added special tokens. tool_choice=auto keeps the tool schema.
    # Match Message.model_dump(exclude_none=True) + generation.render_messages
    # for this fixture's text-only subset. HF's Qwen template expects argument
    # objects, while the OpenAI wire representation is a JSON string.
    normalized = []
    for message in messages:
        normalized_message = copy.deepcopy({key: value for key, value in message.items() if value is not None})
        for call in normalized_message.get('tool_calls', []):
            function = call['function']
            if isinstance(function.get('arguments'), str):
                function['arguments'] = strict_object(function['arguments'])
        normalized.append(normalized_message)
    text = tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False, verbose=False)


def build_cases(tokenizer, document, name, lengths):
    cases = []
    for length in lengths:
        for kind in ('agent_history', 'last_message_edit', 'tool_result_edit', 'special_anchor'):
            fixture_id = digest([VERSION, name, length, kind])
            system = {'role': 'system', 'content': fixture_id + '\n'
                      'Это проверка синтетической истории проекта. Используй только явно названные факты проекта. '
                      'Записи Record — отвлекающие данные. Последнее явное исправление заменяет прежнее значение.'}
            body = document(tokenizer, length, int(fixture_id[:12], 16))
            cut1, cut2 = len(body)//3, 2*len(body)//3
            parts = [body[:cut1], body[cut1:cut2], body[cut2:]]
            facts = ['Факты проекта: code=АЛЬФА-917.\n', '\nФакты проекта: count=23, owner=Ирина.\n',
                     '\nФакты проекта: price=14.\n']
            doc = facts[0]+parts[0]+facts[1]+parts[1]+facts[2]+parts[2]
            tools = None
            if kind == 'agent_history':
                history = [system]
                for i, part in enumerate(parts):
                    history.extend([{'role': 'user', 'content': facts[i]+part},
                                    {'role': 'assistant', 'content': 'Факты приняты.'}])
                target = history + [{'role': 'user', 'content': QUESTION}]
                primers = [history[:3] + [{'role': 'user', 'content': 'Подтверди одним словом: принято.'}],
                           history + [{'role': 'user', 'content': 'Подтверди получение всех частей одним словом.'}]]
                expected = GOLD
            elif kind == 'last_message_edit':
                history = [system, {'role': 'user', 'content': doc},
                           {'role': 'assistant', 'content': 'Готов учитывать изменения.'}]
                primers = [history+[{'role': 'user', 'content': 'Для этого ответа count=23 и price=14. '+QUESTION}]]
                target = history+[{'role': 'user', 'content': 'Исправление: теперь count=29 и price=17. '+QUESTION}]
                expected = EDITED
            elif kind == 'tool_result_edit':
                tools = TOOL
                call = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'fixture_call_01',
                        'type': 'function', 'function': {'name': 'lookup_fixture', 'arguments': '{"key":"source"}'}}]}
                history = [system, {'role': 'user', 'content': doc+'\nПолучим актуальные факты из lookup_fixture.'}, call]
                old_result = {'role': 'tool', 'tool_call_id': 'fixture_call_01', 'content': canonical(GOLD)}
                new_result = dict(old_result, content=canonical(EDITED))
                primers = [history+[old_result, {'role': 'user', 'content': QUESTION}]]
                target = history+[new_result, {'role': 'user', 'content': QUESTION}]
                expected = EDITED
            else:
                tools = TOOL
                history = [system, {'role': 'user', 'content': doc+'\n'
                           'Сейчас вызови lookup_fixture с единственным аргументом key="source". '
                           'После вызова остановись и дождись результата. Результат tool имеет приоритет над прежними фактами. '
                           'Когда результат поступит: '+QUESTION}]
                call = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'fixture_call_anchor',
                        'type': 'function', 'function': {'name': 'lookup_fixture', 'arguments': '{"key":"edited"}'}}]}
                # Keep this as a continuation of the same user turn. A new user
                # turn makes Qwen's template strip the old empty <think> prefix,
                # preventing reuse of the generated tool opener by construction.
                target = history+[call, {'role': 'tool', 'tool_call_id': 'fixture_call_anchor', 'content': canonical(EDITED)}]
                primers = [history]
                expected = EDITED
            target_ids = render_ids(tokenizer, target, tools)
            primer_ids = [render_ids(tokenizer, primer, tools) for primer in primers]
            lcp = max(common_prefix(target_ids, ids) for ids in primer_ids)
            anchor_ids = tokenizer.encode('<tool_call>', add_special_tokens=False)
            anchor = tool_anchor_boundary(target_ids, primer_ids[0], anchor_ids) if kind == 'special_anchor' else None
            # Token arrays/text stay in memory. Fixed fixture and gold hashes are
            # identical in both phases; no namespace trick changes the target.
            cases.append({'id': fixture_id, 'kind': kind, 'length': length, 'target': target,
                          'primers': primers, 'tools': tools, 'expected': dict(expected),
                          'manifest': {'case_id': fixture_id, 'kind': kind, 'document_tokens': length,
                              'prompt_tokens': len(target_ids), 'prompt_tokens_sha256': digest(target_ids),
                              'target_sha256': digest([target, tools]), 'gold_sha256': digest(expected),
                              'primer_sha256': [digest([p, tools]) for p in primers],
                              'primer_prompt_tokens': [len(ids) for ids in primer_ids],
                              'shared_prompt_tokens': lcp, 'anchor_boundary_tokens': anchor}})
            cases[-1]['manifest']['generation_prefix_aligned'] = (
                common_prefix(target_ids, primer_ids[0]) == len(primer_ids[0]) if kind == 'special_anchor' else None)
    return cases


def numeric_tree(value, depth=0):
    """Explicitly exclude arbitrary strings from metrics and nested error data."""
    if depth > 6:
        return None
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: cleaned for key, val in value.items()
                if isinstance(key, str) and re.fullmatch(r'[A-Za-z0-9_]{1,80}', key)
                and (cleaned := numeric_tree(val, depth+1)) is not None}
    if isinstance(value, list) and len(value) < 128:
        return [numeric_tree(item, depth+1) for item in value]
    return None


def result_metrics(result):
    allowed = ('status', 'usage', 'ttft_content_s', 'ttft_reasoning_s', 'ttft_generated_s',
               'elapsed_s', 'decode_client_tps', 'decode_ack_tps', 'cached_prompt_tokens',
               'peaks', 'counter_deltas', 'engine')
    metrics = {key: numeric_tree(result[key]) for key in allowed if key in result}
    metrics['engine_metrics_joined'] = result.get('engine_metrics_status') == 'joined'
    metrics['finished_normally'] = result.get('finish_reason') in ('stop', 'tool_calls')
    return metrics


def cached_tokens(result):
    value = result.get('cached_prompt_tokens')
    if type(value) is not int:
        value = result.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens')
    return value if type(value) is int and value >= 0 else None


def run_identity(receipt):
    config = receipt.get('configuration', {})
    run_id = receipt.get('run_id')
    if not isinstance(run_id, str) or not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise ValueError('missing_run_receipt')
    if not isinstance(receipt.get('source_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', receipt['source_sha256']):
        raise ValueError('missing_source_receipt')
    if config.get('env', {}).get('QWEN36_TEACHER_FIXTURE'):
        raise ValueError('teacher_forcing_cannot_qualify_functional_outputs')
    return {'run_id_sha256': digest(run_id), 'profile_sha256': digest(receipt.get('profile')),
            'configuration_sha256': digest(config), 'source_sha256': receipt.get('source_sha256'),
            'checkpoint_sha256': digest([receipt.get('checkpoint'), receipt.get('checkpoint_revision')]),
            'packages_sha256': digest(receipt.get('packages', {})),
            'context': int(config.get('context', 65536)), 'chunk': int(config.get('chunk', 2048)),
            'prefix_margin': max(1, int(config.get('env', {}).get('QWEN36_PREFIX_MARGIN', 1))),
            'anchor_enabled': '--enable-special-token-ckpt' in config.get('flags', [])}


def same_runtime(left, right):
    keys = ('profile_sha256', 'configuration_sha256', 'source_sha256', 'checkpoint_sha256', 'packages_sha256')
    return all(left.get(key) == right.get(key) for key in keys)


def send(messages, tools=None, force_tool=False):
    import bench
    if tools is None:
        return bench.generate(messages, MAXIMUM, SEED, temperature=0)
    import requests
    payload = {'model': 'qwen36-35b', 'messages': messages, 'tools': tools,
               'tool_choice': {'type': 'function', 'function': {'name': 'lookup_fixture'}} if force_tool else 'auto',
               'max_tokens': MAXIMUM, 'temperature': 0, 'top_p': 1, 'seed': SEED,
               'chat_template_kwargs': {'enable_thinking': False}, 'stream': False}
    started = time.perf_counter()
    with requests.post('http://127.0.0.1:1919/v1/chat/completions', json=payload,
                       headers=bench.headers(), timeout=(5, 900)) as response:
        result = {'status': response.status_code, 'elapsed_s': time.perf_counter()-started,
                  'request_id': response.headers.get('x-qwen36-request-id')}
        if response.status_code != 200:
            return result, ''  # Never read/log a possibly private HTTP error body.
        body = response.json()
        choice = body.get('choices', [{}])[0]
        message = choice.get('message', {})
        result.update(usage=body.get('usage', {}), finish_reason=choice.get('finish_reason'))
        result['tool_call_count'] = len(message.get('tool_calls', []))
        bench.attach_metrics(result)
        if force_tool:
            calls = message.get('tool_calls', [])
            valid = len(calls) == 1
            try:
                call = calls[0]
                valid = valid and isinstance(call.get('id'), str) and bool(call['id'])
                valid = valid and call['function']['name'] == 'lookup_fixture'
                valid = valid and strict_object(call['function']['arguments']) == {'key': 'source'}
            except (IndexError, KeyError, TypeError, ValueError):
                valid = False
            result['tool_call_valid'] = bool(valid)
            result['tool_call_count'] = len(calls)
        return result, message.get('content') or ''


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def phase_gate(phase, row, baseline=None, require_anchor=False):
    functional = row['verdict']['pass'] and row['status'] == 200
    covered = row['fresh_zero_cache'] if phase == 'fresh' else row['prefix_hit_observed']
    if phase == 'cached':
        functional = functional and baseline['gate_pass'] and all(p['pass'] for p in row['primers'])
        if require_anchor and row['kind'] == 'special_anchor':
            covered = covered and row['anchor_observed']
    return bool(functional and covered and row['token_count_matches'] and row['request_run_matches']
                and row['no_unexpected_tools'] and row['metrics']['engine_metrics_joined'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('fresh', 'cached'))
    parser.add_argument('--name', default='prefix-fixed-v1')
    parser.add_argument('--report-prefix', type=Path)
    parser.add_argument('--lengths', default='16384,60000')
    parser.add_argument('--require-anchor', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.name):
        parser.error('Invalid name')
    if not args.prepare_only and args.phase is None:
        parser.error('--phase is required')
    lengths = [int(value) for value in args.lengths.split(',')]
    if not lengths or len(set(lengths)) != len(lengths) or any(n < 512 or n > 250000 for n in lengths):
        parser.error('Distinct lengths must be in 512..250000')
    import bench
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get('QWEN36_MODEL_DIR','/path/to/LLM/qwen36-api/model'), local_files_only=True)
    cases = build_cases(tokenizer, bench.document, args.name, lengths)
    manifest = [case['manifest'] for case in cases]
    prefix = args.report_prefix or bench.ROOT/'reports'/args.name
    prepared = {'fixture_version': VERSION, 'name_sha256': digest(args.name), 'manifest': manifest,
                'manifest_sha256': digest(manifest), 'model_requests': 0,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'bench_sha256': hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest() if hasattr(bench, '__file__') else None,
                'sampling': {'temperature': 0, 'top_p': 1, 'seed': SEED,
                             'thinking': False, 'max_tokens': MAXIMUM}}
    if args.prepare_only:
        write_report(Path(str(prefix)+'-prepared.json'), prepared)
        print(canonical({'prepared': True, 'cases': len(cases), 'manifest_sha256': digest(manifest), 'model_requests': 0}))
        return
    identity = run_identity(bench.run_receipt())
    if any(case['manifest']['prompt_tokens']+MAXIMUM > identity['context'] for case in cases):
        raise ValueError('context_budget_exceeded_before_request')
    if any(case['kind'] == 'special_anchor' and not case['manifest']['generation_prefix_aligned'] for case in cases):
        raise ValueError('template_does_not_preserve_generated_anchor_prefix')
    path = Path(str(prefix)+'-'+args.phase+'.json')
    if path.exists():
        raise ValueError('report_already_exists_use_new_name_and_clean_run')
    baseline = None
    if args.phase == 'cached':
        baseline = json.loads(Path(str(prefix)+'-fresh.json').read_text())
        if (baseline.get('manifest_sha256') != digest(manifest) or not baseline.get('completed')
                or baseline.get('script_sha256') != prepared['script_sha256']
                or baseline.get('bench_sha256') != prepared['bench_sha256']
                or not same_runtime(baseline['run'], identity)
                or baseline['run']['run_id_sha256'] == identity['run_id_sha256']):
            raise ValueError('fresh_receipt_mismatch_or_no_restart')
    report = dict(prepared, phase=args.phase, run=identity, require_anchor=args.require_anchor,
                  completed=False, gate_pass=False, rows=[],
                  anchor_coverage_expected=identity['anchor_enabled'])
    write_report(path, report)
    for index, case in enumerate(cases):
        # Refuse replacement of the serving run halfway through either phase.
        if run_identity(bench.run_receipt()) != identity:
            raise ValueError('active_run_changed')
        primers = []
        if args.phase == 'cached':
            for messages in case['primers']:
                result, _text = send(messages, case['tools'], case['kind'] == 'special_anchor')
                ok = result.get('status') == 200 and result.get('finish_reason') in ('stop', 'tool_calls')
                if case['kind'] == 'special_anchor':
                    ok = ok and result.get('tool_call_valid', False)
                primers.append({'pass': bool(ok), 'metrics': result_metrics(result),
                                'tool_call_valid': result.get('tool_call_valid'),
                                'tool_call_count': result.get('tool_call_count')})
                report['model_requests'] += 1
        result, text = send(case['target'], case['tools'])
        report['model_requests'] += 1
        cached = cached_tokens(result)
        # A substantial observed hit is mandatory, not inferred from TTFT. The
        # Shared boundary minus one prefill chunk, the configured margin, and
        # one 64-token state block tolerates the documented carry/checkpoint
        # granularity; it does not relax any functional or numerical criterion.
        minimum_hit = max(64, case['manifest']['shared_prompt_tokens']-identity['chunk']
                          -identity['prefix_margin']-64)
        anchor = case['manifest']['anchor_boundary_tokens']
        row = {'case_id': case['id'], 'kind': case['kind'], 'status': result.get('status', 0),
               'verdict': verdict(text, case['expected']), 'content_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'metrics': result_metrics(result), 'cached_prompt_tokens': cached,
               'fresh_zero_cache': cached == 0,
               'prefix_hit_observed': cached is not None and cached >= minimum_hit,
               'minimum_coverage_tokens': minimum_hit,
               'anchor_observed': bool(identity['anchor_enabled'] and anchor is not None
                                       and cached is not None and cached >= anchor),
               'token_count_matches': result.get('usage', {}).get('prompt_tokens') == case['manifest']['prompt_tokens'],
               'request_run_matches': digest(result.get('engine', {}).get('run_id')) == identity['run_id_sha256'],
               'no_unexpected_tools': result.get('tool_call_count', 0) == 0,
               'primers': primers}
        reference = baseline['rows'][index] if baseline is not None else None
        if reference:
            if reference['case_id'] != row['case_id']:
                raise ValueError('case_order_mismatch')
            row['fresh_cached_output_sha_equal'] = reference['content_sha256'] == row['content_sha256']
            row['fresh_cached_parsed_sha_equal'] = reference['verdict'].get('parsed_sha256') == row['verdict'].get('parsed_sha256')
            row['functional_regression'] = reference['verdict']['pass'] and not row['verdict']['pass']
        row['gate_pass'] = phase_gate(args.phase, row, reference, args.require_anchor)
        report['rows'].append(row)
        write_report(path, report)
        print(canonical({'case_id': row['case_id'], 'phase': args.phase, 'gate_pass': row['gate_pass'],
                         'functional_pass': row['verdict']['pass'], 'cached_prompt_tokens': cached}), flush=True)
    report['completed'] = True
    report['gate_pass'] = all(row['gate_pass'] for row in report['rows'])
    report['anchor_gate_pass'] = identity['anchor_enabled'] and all(
        row['anchor_observed'] for row in report['rows'] if row['kind'] == 'special_anchor') if args.phase == 'cached' else None
    write_report(path, report)
    print(canonical({'completed': True, 'gate_pass': report['gate_pass'], 'cases': len(cases),
                     'anchor_enabled': identity['anchor_enabled'], 'model_requests': report['model_requests']}))
    if not report['gate_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Exception messages/HTTP bodies could contain request text. Emit type only.
        print(canonical({'completed': False, 'exception_type': type(exc).__name__}), flush=True)
        raise SystemExit(2) from None
