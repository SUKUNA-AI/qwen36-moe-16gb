"""Cancel fresh synthetic prefill, then prove the inference slot works again."""
import os
import argparse, asyncio, contextlib, json, time
import httpx
from transformers import AutoTokenizer
from bench import ROOT, document, headers, run_receipt


async def check(length, name):
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get('QWEN36_MODEL_DIR','/path/to/LLM/qwen36-api/model'), local_files_only=True)
    body = document(tokenizer, length, 9893431)
    payload = {'model':'qwen36-35b', 'messages':[{'role':'user','content':body}],
               'max_tokens':128, 'temperature':0, 'stream':False,
               'chat_template_kwargs':{'enable_thinking':False}}
    auth = headers(); base = 'http://127.0.0.1:1919'
    async with httpx.AsyncClient(timeout=30) as client:
        task = asyncio.create_task(client.post(base+'/v1/chat/completions', headers=auth, json=payload))
        observed = False
        try:
            for _ in range(50):
                await asyncio.sleep(.1)
                response = await client.get(base+'/v1/stats', headers=auth)
                if response.status_code == 200 and response.json().get('requests',{}).get('active') == 1:
                    observed = True; break
                if task.done(): break
            await asyncio.sleep(2)
            pending_at_cancel = not task.done()
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError): await task
        started = time.monotonic(); statuses = []; correct = False
        small = {**payload, 'messages':[{'role':'user','content':'Ответь только числом: 17 + 25.'}], 'max_tokens':32}
        for _ in range(100):
            response = await client.post(base+'/v1/chat/completions', headers=auth, json=small)
            statuses.append(response.status_code)
            if response.status_code == 200:
                correct = response.json()['choices'][0]['message']['content'].strip() == '42'
                break
            if response.status_code != 429: break
            await asyncio.sleep(.1)
        elapsed = time.monotonic()-started
    report = {'run':run_receipt(), 'length':length, 'observed_active':observed,
              'request_pending_at_cancel':pending_at_cancel, 'statuses':statuses,
              'followup_correct':correct, 'recovery_s':elapsed,
              'pass':observed and pending_at_cancel and correct and elapsed < 10}
    (ROOT/'reports'/f'{name}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'run'}))
    assert report['pass']


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--length',type=int,required=True)
    parser.add_argument('--name',required=True);args=parser.parse_args()
    asyncio.run(check(args.length,args.name))
