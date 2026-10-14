import ast
import contextlib
from types import SimpleNamespace as S
import unittest
from cpu_control_flow import methods
from repo_paths import FREETOKEN
SOURCE=FREETOKEN/'python/freetoken/scheduler/scheduler.py'

def trace(mutate=None, job=False):
    events=[]
    class Stream:
        def __init__(self,name):self.name=name
        def wait_stream(self,other):events.append(f'{self.name}<-{other.name}')
    sched=Stream('scheduler');engine=Stream('engine')
    def mark(name,result=None):
        def call(*a,**kw):events.append(name);return result
        return call
    fake=S(_pending_rebuild=None,stream=sched,engine=S(stream=engine),engine_stream_ctx=contextlib.nullcontext(),
           prefill_manager=S(runnable=True,layer_major_job=S(step=mark('step','data')) if job else None),
           decode_manager=S(runnable=False), receive_msg=lambda **kw:[],
           _schedule_next_batch=mark('schedule',S(batch='batch')),_restore_linear_states=mark('restore'),
           _forward_or_layer_major=mark('forward','data'),_process_last_data=mark('commit'),_flush_abort_acks=mark('acks'))
    methods(SOURCE,'Scheduler',{'overlap_loop'}, {},mutate)['overlap_loop'](fake,'previous')
    return events

class Ordering(unittest.TestCase):
    def test_actual_decode_and_commit_flow(self):
        self.assertEqual(trace(),['scheduler<-engine','schedule','engine<-scheduler','restore','forward','scheduler<-engine','commit','acks'])
    def test_layer_major_flow(self):
        self.assertEqual(trace(job=True),['scheduler<-engine','schedule','engine<-scheduler','step','scheduler<-engine','commit','acks'])
    def test_mutants_missing_or_reordered_commit_wait_fail_trace(self):
        good=trace()
        def mutate(tree):
            body=tree.body[0].body
            wait=next(n for n in reversed(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
                      isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='wait_stream')
            body.remove(wait)
        self.assertNotEqual(trace(mutate),good)
        def reorder(tree):
            body=tree.body[0].body
            waits=[i for i,n in enumerate(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
                   isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='wait_stream']
            i=waits[-1];body[i],body[i+1]=body[i+1],body[i]
        self.assertNotEqual(trace(reorder),good)
if __name__=='__main__':unittest.main(verbosity=2)
