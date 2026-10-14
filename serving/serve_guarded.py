import asyncio, logging, os
os.environ.pop('FREETOKEN_API_LOG_DIR',None)
from guard import APIGuard, CLEANUP_SECONDS, start_abort
from ingress import Ingress
from request_metrics import CURRENT, RequestMetrics, install_native_observers
from placement import install
install()

def main(run_id=None):
    # One generation ID inherited by every worker; request UIDs restart at zero.
    # Always replace inherited values. The isolated launcher can supply the UUID
    # it has just generated so its immutable source receipt identifies this run.
    import uuid
    os.environ['QWEN36_RUN_ID'] = uuid.UUID(run_id).hex if run_id is not None else uuid.uuid4().hex
    from freetoken.server import api_server, openai_api
    from freetoken.cli import main as ft_main
    api_server.app.add_middleware(APIGuard,state_getter=api_server.get_global_state)
    api_server.app.add_middleware(RequestMetrics,state_getter=api_server.get_global_state)
    api_server.app.add_middleware(Ingress)
    install_native_observers(api_server,openai_api)
    original_uvicorn_run=api_server.uvicorn.run
    def private_run(*args,**kwargs):
        kwargs['access_log']=False
        kwargs['timeout_keep_alive']=5
        return original_uvicorn_run(*args,**kwargs)
    api_server.uvicorn.run=private_run
    original_submit=openai_api.submit_generation
    async def tracked_submit(spec,state):
        uid=await original_submit(spec,state)
        record=CURRENT.get()
        if record is not None:
            record.engine_uid=uid
            record.row['engine_uid']=uid
        return uid
    openai_api.submit_generation=tracked_submit
    original=openai_api.generate_full
    async def cancellation_safe_full(uid,spec,state,**kwargs):
        try:
            result=await original(uid,spec,state,**kwargs)
            metrics=CURRENT.get()
            if metrics is not None: metrics.terminal(result)
            return result
        except asyncio.CancelledError:
            # A Starlette/AnyIO cancellation scope may cancel every awaited
            # checkpoint. Dispatch in its own task so abort reaches the worker.
            task=start_abort(state,uid)
            try: await asyncio.wait_for(asyncio.shield(task),CLEANUP_SECONDS)
            except Exception: pass
            raise
    openai_api.generate_full=cancellation_safe_full
    logging.getLogger('uvicorn.access').disabled=True
    return ft_main()

if __name__=='__main__': main()
