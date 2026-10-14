"""Owner-only, local CLI view of numeric request metrics. Never contacts inference."""
import collections,json,pathlib,statistics,subprocess
root=pathlib.Path('/home/USER/qwen36-api/logs')

def tail(path,limit=50):
    if not path.exists(): return []
    with path.open() as stream: lines=collections.deque(stream,maxlen=limit)
    result=[]
    for line in lines:
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass # A concurrently written final line can be partial.
    return result

rows=tail(root/'requests.jsonl',1000)
session=rows[-1]['session_id'] if rows else None
recent=[x for x in rows if x['kind']=='generation' and x['session_id']==session][-20:]
state=next(iter(tail(root/'state.jsonl',1)),None)
if state and state['session_id']!=session:state=None
good=[x for x in recent if x['outcome']=='completed' and x['counts_final']]
def mean(key):
    values=[x[key] for x in good if x.get(key) is not None]
    return round(statistics.mean(values),3) if values else None
print(json.dumps({'source':'owner-only files; historical data, not a readiness check',
                  'directory':str(root),'last_session_id':session,
                  'service_state':subprocess.run(['systemctl','--user','is-active','qwen36-inference.service'],capture_output=True,text=True).stdout.strip(),
                  'averages_note':'Last 20 generation attempts in the last session; completed requests of different lengths are mixed. Not a benchmark.',
                  'recent_generation_count':len(recent),
                  'outcomes':dict(collections.Counter(x['outcome'] for x in recent)),
                  'mean_ttft_generated_s':mean('ttft_generated_s'),
                  'mean_ttft_content_s':mean('ttft_content_s'),
                  'mean_decode_ack_tps':mean('decode_ack_tps'),
                  'last_state':state,
                  'last_requests':[{k:r.get(k) for k in ('ts','request_id','status','outcome','prompt_tokens','completion_tokens','cached_prompt_tokens','ttft_generated_s','ttft_content_s','duration_s','decode_ack_tps','counts_final')} for r in recent[-5:]]},ensure_ascii=False,indent=2))
