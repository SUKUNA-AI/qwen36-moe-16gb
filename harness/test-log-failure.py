"""Bounded telemetry queue, full filesystem and close semantics without GPU."""
import json,pathlib,tempfile,threading,time
from unittest.mock import patch
from request_metrics import BufferedNumericLog,NumericLog

with tempfile.TemporaryDirectory(prefix='qwen36-log-failure-') as temp:
    entered=threading.Event();release=threading.Event()
    def blocked(self,*args):entered.set();release.wait(3)
    with patch.object(NumericLog,'write',blocked):
        log=BufferedNumericLog(pathlib.Path(temp)/'bounded',capacity=2)
        log.write('requests.jsonl',{'n':0});assert entered.wait(2)
        for i in range(20):log.write('requests.jsonl',{'n':i+1})
        assert log.pending.qsize()==2 and log.dropped==18
        release.set();assert log.flush(3);log.close();assert not log.worker.is_alive()
    log=BufferedNumericLog(pathlib.Path(temp)/'full')
    with patch('request_metrics.os.open',side_effect=OSError(28,'No space left on device')):
        for i in range(3):log.write('requests.jsonl',{'n':i})
        assert log.flush(3);assert log.failed_writes==3
    log.write('requests.jsonl',{'recovered':True});assert log.flush(3);assert not log.failed
    log.close();assert not log.worker.is_alive()
    print(json.dumps({'bounded_queue':True,'overflow_count':18,'disk_full_isolated':True,'writer_recovers':True,'threads_joined':True}))
