"""Print the FP8 serving run_ids used by the quality decision (teacher, functional, eval), taken from the reports."""
import json
R = '/home/USER/qwen36-api/optimization/reports/'
ids = [r['run_id'] for r in map(json.loads, filter(str.strip, open(R + 'teacher.jsonl')))
       if r.get('profile') == 'teacher-main-release-fp8kv']
ids = ids[-1:]
ids.append(json.load(open(R + 'functional-main-release-fp8kv.json'))['run']['run_id'])
ids.append(json.load(open(R + 'fp8kv-eval-fp8-eval-v1.json'))['run']['run_id'])
print(','.join(dict.fromkeys(ids)))
