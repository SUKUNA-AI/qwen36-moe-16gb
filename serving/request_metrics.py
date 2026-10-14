"""Private numeric telemetry for the pinned FreeToken API; no prompts/outputs/headers.

Token counts come from scheduler acknowledgements and terminal generation events,
including streams without stream_options.include_usage. Cancellation counts are
explicitly partial. This module never writes exception messages or arbitrary paths.
"""
import asyncio
import contextvars
import contextlib
import datetime
import json
import os
import pathlib
import queue
import threading
import time
import uuid

ROOT = pathlib.Path(os.environ.get('QWEN36_DATA_ROOT','/home/USER/qwen36-api')) / 'logs'
CURRENT = contextvars.ContextVar('qwen36_request_metrics', default=None)
ENGINE_PROGRESS = time.monotonic()
KINDS = {
    ('POST', '/v1/chat/completions'): 'generation',
    ('GET', '/v1/models'): 'models',
    ('GET', '/health'): 'health',
    ('GET', '/v1/stats'): 'local_stats',
    ('GET', '/v1/cache/status'): 'local_cache_status',
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class NumericLog:
    def __init__(self, root=ROOT, max_bytes=10 * 1024 * 1024, backups=5):
        self.root = pathlib.Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.max_bytes, self.backups = max_bytes, backups
        self.lock = threading.Lock()
        self.failed = False
        self.failed_writes = 0

    def write(self, name, row):
        # The caller supplies a closed schema; never pass request/response objects here.
        try:
            line = json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n'
            with self.lock:
                path = self.root / name
                limit = self.max_bytes if name == 'requests.jsonl' else 2 * 1024 * 1024
                if path.exists() and path.stat().st_size + len(line) > limit:
                    for i in range(self.backups, 0, -1):
                        old = path if i == 1 else self.root / (name + '.' + str(i - 1))
                        new = self.root / (name + '.' + str(i))
                        if old.exists(): os.replace(old, new)
                fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                try:
                    with os.fdopen(fd, 'a', encoding='utf-8') as out: out.write(line)
                finally:
                    path.chmod(0o600)
            self.failed = False
        except (OSError, ValueError, TypeError):
            # Telemetry failure must not turn a valid inference request into an error.
            if not self.failed: print('QWEN36_METRICS_WRITE_FAILED', flush=True)
            self.failed = True
            self.failed_writes += 1


class Resources:
    def __init__(self):
        self.gpu = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None
        cg = None
        for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'): cg = line[3:]
        self.memory_path = pathlib.Path('/sys/fs/cgroup' + cg + '/memory.current') if cg else None

    def close(self):
        if self.nvml is not None:
            with contextlib.suppress(Exception): self.nvml.nvmlShutdown()
            self.nvml = None
            self.gpu = None

    def snapshot(self):
        row = {'ts': utc_now()}
        try:
            row['service_cgroup_memory_bytes'] = int(self.memory_path.read_text())
        except (OSError, AttributeError, ValueError): pass
        try:
            stat=dict(line.split() for line in self.memory_path.with_name('cpu.stat').read_text().splitlines())
            row['service_cpu_usage_usec']=int(stat['usage_usec'])
        except (OSError, AttributeError, ValueError, KeyError):pass
        try:
            stat = dict(line.split() for line in self.memory_path.with_name('memory.stat').read_text().splitlines())
            for key in ('anon','file','shmem','kernel','pagetables','sock','file_mapped','unevictable'):
                row['service_memory_' + key + '_bytes'] = int(stat.get(key,0))
            # cgroup file includes tmpfs/shared expert banks. This estimate
            # separates them from disk-backed file pages; it is not free RAM.
            row['service_memory_file_cache_estimate_bytes'] = max(0,int(stat.get('file',0))-int(stat.get('shmem',0)))
            row['service_memory_peak_bytes'] = int(self.memory_path.with_name('memory.peak').read_text())
        except (OSError, AttributeError, ValueError): pass
        try:
            for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'): row['wsl_mem_available_bytes'] = int(line.split()[1]) * 1024
                elif line.startswith('SwapFree:'): row['wsl_swap_free_bytes'] = int(line.split()[1]) * 1024
        except (OSError, ValueError): pass
        if self.gpu is not None:
            for name, getter in [
                ('gpu_memory', lambda: self.nvml.nvmlDeviceGetMemoryInfo(self.gpu)),
                ('gpu_utilization', lambda: self.nvml.nvmlDeviceGetUtilizationRates(self.gpu)),
                ('gpu_temperature_c', lambda: self.nvml.nvmlDeviceGetTemperature(self.gpu, self.nvml.NVML_TEMPERATURE_GPU)),
                ('gpu_power_mw', lambda: self.nvml.nvmlDeviceGetPowerUsage(self.gpu)),
                ('gpu_energy_mj', lambda: self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu)),
            ]:
                try:
                    value = getter()
                    if name == 'gpu_memory':
                        row.update(gpu_total_used_bytes=int(value.used), gpu_total_bytes=int(value.total), gpu_free_bytes=int(value.free))
                    elif name == 'gpu_utilization': row['gpu_utilization_pct'] = int(value.gpu)
                    else: row[name] = int(value)
                except Exception: pass  # Some NVML functions are unsupported under WSL/WDDM.
        return row


class BufferedNumericLog(NumericLog):
    def __init__(self, *args, capacity=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.closed = False
        self.enqueue_lock = threading.Lock()
        self.worker = threading.Thread(target=self.drain, name='qwen36-log-writer', daemon=True)
        self.worker.start()

    def write(self, name, row):
        # close() cannot let the writer exit between this check and enqueue.
        with self.enqueue_lock:
            if self.closed:
                self.dropped += 1
                return
            try: self.pending.put_nowait((name, row))
            except queue.Full: self.dropped += 1

    def drain(self):
        while not self.closed or not self.pending.empty():
            try:item = self.pending.get(timeout=.2)
            except queue.Empty:continue
            try:
                super().write(*item)
            finally: self.pending.task_done()

    def flush(self, timeout=10):
        deadline=time.monotonic()+timeout
        with self.pending.all_tasks_done:
            while self.pending.unfinished_tasks:
                remaining=deadline-time.monotonic()
                if remaining<=0:return False
                self.pending.all_tasks_done.wait(remaining)
        return True

    def close(self):
        with self.enqueue_lock:
            if self.closed: return
            self.closed = True
        self.worker.join(timeout=10)


class CachedResources:
    def __init__(self, source):
        self.source = source
        self.latest = source.snapshot()

    def refresh(self):
        self.latest = self.source.snapshot()
        return dict(self.latest)

    def snapshot(self): return dict(self.latest)


class RequestRecord:
    def __init__(self, scope, session_id, resources):
        self.start = time.monotonic()
        self.row = {
            'schema': 2, 'ts': utc_now(), 'request_id': uuid.uuid4().hex,
            'run_id': os.environ.get('QWEN36_RUN_ID'),
            'model': 'qwen36-35b', 'context_limit': int(os.environ.get('QWEN36_CONTEXT', '65536')),
            'session_id': session_id, 'kind': KINDS.get((scope['method'], scope['path']), 'blocked_route'),
            'stream': None, 'status': None, 'outcome': 'completed',
            'prompt_tokens': None, 'completion_tokens': None, 'cached_prompt_tokens': None,
            'counts_final': False, 'ttft_generated_s': None, 'ttft_content_s': None,
            'ttft_reasoning_s': None, 'ttft_tool_s': None, 'decode_ack_tps': None,
            'prefill_tps': None, 'resource_samples': 0, 'peaks': {},
        }
        self.resources = resources
        self.ack_prompt = self.ack_completion = self.ack_cached = 0
        self.first_ack = self.last_ack = None
        self.first_ack_tokens = 0
        self.disconnected = False
        self.counter_first = {}
        self.counter_last = {}

    def configure(self, payload):
        self.row['stream'] = bool(payload.get('stream', False))
        # Only finite numeric sampling values and a boolean thinking flag are retained.
        self.row['sampling'] = {}
        for name in ('temperature', 'top_p', 'top_k', 'max_tokens', 'max_completion_tokens'):
            value = payload.get(name)
            if type(value) in (int, float) and abs(value) < 1e9:
                self.row['sampling'][name] = value
        if type(payload.get('seed')) is int and 0<=payload['seed']<2**63:
            self.row['sampling']['seed']=payload['seed']
        ctk = payload.get('chat_template_kwargs')
        if isinstance(ctk, dict) and type(ctk.get('enable_thinking')) is bool:
            self.row['sampling']['enable_thinking'] = ctk['enable_thinking']

    def sample(self, values):
        self.row['resource_samples'] += 1
        for key in ('gpu_total_used_bytes', 'service_cgroup_memory_bytes', 'gpu_utilization_pct', 'gpu_temperature_c',
                    'gpu_power_mw','service_memory_anon_bytes','service_memory_file_bytes','service_memory_kernel_bytes',
                    'service_memory_shmem_bytes','service_memory_file_cache_estimate_bytes',
                    'service_memory_pagetables_bytes','service_memory_file_mapped_bytes','service_memory_unevictable_bytes'):
            if key in values: self.row['peaks'][key] = max(self.row['peaks'].get(key, 0), values[key])
        for key in ('service_cpu_usage_usec','gpu_energy_mj'):
            if key in values:
                self.counter_first.setdefault(key,values[key])
                self.counter_last[key]=values[key]
        for key in ('wsl_mem_available_bytes', 'gpu_free_bytes', 'wsl_swap_free_bytes'):
            if key in values: self.row['peaks']['min_' + key] = min(self.row['peaks'].get('min_' + key, values[key]), values[key])

    def acknowledge(self, ack):
        self.ack_prompt += int(getattr(ack, 'prompt_tokens_delta', 0))
        self.ack_cached += int(getattr(ack, 'cached_tokens', 0))
        delta = int(getattr(ack, 'completion_tokens_delta', 0))
        self.ack_completion += delta
        if delta > 0:
            now = time.monotonic()
            if self.first_ack is None:
                self.first_ack, self.first_ack_tokens = now, delta
                self.row['ttft_generated_s'] = now - self.start
            self.last_ack = now
        if getattr(ack, 'error', None): self.row['outcome'] = 'engine_error'

    def semantic(self, event):
        # Inspect only event type / whether text exists; text and tool arguments are never stored.
        name = type(event).__name__
        field = {'ContentDelta': 'ttft_content_s', 'ReasoningDelta': 'ttft_reasoning_s',
                 'ToolCallStart': 'ttft_tool_s', 'ToolCallsDelta': 'ttft_tool_s'}.get(name)
        if field and (name not in ('ContentDelta', 'ReasoningDelta') or getattr(event, 'text', '')):
            if self.row[field] is None: self.row[field] = time.monotonic() - self.start
        if name == 'GenDone': self.terminal(event)

    def terminal(self, result):
        self.row.update(prompt_tokens=int(result.prompt_tokens), completion_tokens=int(result.completion_tokens),
                        cached_prompt_tokens=int(result.cached_tokens), counts_final=True)
        reason = result.finish_reason
        self.row['finish_reason'] = reason if reason in ('stop', 'length', 'tool_calls', 'content_filter') else 'other'

    def finish(self, scope):
        self.sample(self.resources.snapshot())
        self.row['duration_s'] = time.monotonic() - self.start
        self.row['counter_deltas']={key:max(0,value-self.counter_first[key]) for key,value in self.counter_last.items()}
        self.row['resource_sampling_period_s']=1
        if not self.row['counts_final'] and self.ack_completion + self.ack_prompt:
            self.row.update(prompt_tokens=self.ack_prompt, completion_tokens=self.ack_completion, cached_prompt_tokens=self.ack_cached)
        if self.first_ack is not None and self.last_ack > self.first_ack:
            self.row['decode_ack_tps'] = (self.ack_completion - self.first_ack_tokens) / (self.last_ack - self.first_ack)
        if scope.get('qwen36.outcome') == 'timeout': self.row['outcome'] = 'timeout'
        elif self.disconnected and not self.row['counts_final']: self.row['outcome'] = 'cancelled'
        elif self.row['status'] is not None and self.row['status'] >= 400: self.row['outcome'] = 'rejected'
        self.row['counts_note'] = 'terminal_engine_totals' if self.row['counts_final'] else 'partial_observed_acknowledgements_or_unavailable'
        return self.row


class RequestMetrics:
    def __init__(self, app, state_getter=None, log_root=ROOT, resources=None, monitor=True):
        self.stall_seconds = float(os.environ.get('QWEN36_STALL_SECONDS','300'))
        if not 0 < self.stall_seconds < 86400: raise ValueError('Invalid stall deadline')
        self.app, self.state_getter = app, state_getter
        self.writer = BufferedNumericLog(log_root)
        self.resources = CachedResources(resources or Resources())
        self.session_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.active_records = set()
        self.stop = threading.Event()
        self.resource_failures = 0
        self.closed = False
        self.close_task = None
        self.thread = threading.Thread(target=self.monitor, name='qwen36-numeric-metrics', daemon=True)
        if monitor: self.thread.start()

    def monitor(self):
        # One NVML sample/s; state history every 10s, plus request start/end samples.
        tick = 0
        busy_since = None
        while not self.stop.wait(1):
            try:
                values = self.resources.refresh()
            except Exception:
                # A failed telemetry sample must not disable the independent
                # worker-stall detector or make HTTP requests fail.
                self.resource_failures += 1
                values = self.resources.snapshot()
                values['resource_sample_stale'] = True
            with self.lock:
                if not values.get('resource_sample_stale'):
                    for record in self.active_records: record.sample(values)
            tick += 1
            try:
                active=self.state_getter().stats.active
                if active:
                    if busy_since is None:busy_since=time.monotonic()
                    deadline=self.stall_seconds
                    if time.monotonic()-max(busy_since,ENGINE_PROGRESS)>deadline:
                        self.writer.write('state.jsonl',{'ts':utc_now(),'event':'engine_stall_shutdown','deadline_s':deadline})
                        print('QWEN36_ENGINE_STALL_SHUTDOWN',flush=True)
                        import signal
                        os.kill(os.getpid(),signal.SIGTERM)
                        return
                else:busy_since=None
            except (AssertionError,AttributeError,TypeError):pass
            from guard import PROTOCOL_FAILURE_AT
            if PROTOCOL_FAILURE_AT is not None and time.monotonic()-PROTOCOL_FAILURE_AT>self.stall_seconds:
                self.writer.write('state.jsonl',{'ts':utc_now(),'event':'protocol_stall_shutdown'})
                print('QWEN36_PROTOCOL_STALL_SHUTDOWN',flush=True)
                import signal
                os.kill(os.getpid(),signal.SIGTERM)
                return
            if tick % 10 == 0:
                from ingress import COUNTERS
                values.update(schema=2, session_id=self.session_id, model='qwen36-35b',
                              ingress=dict(COUNTERS), metrics_dropped=self.writer.dropped,
                              metrics_failed_writes=self.writer.failed_writes,
                              resource_sample_failures=self.resource_failures,
                              metrics_queue_depth=self.writer.pending.qsize(),
                              kv_dtype=os.environ.get('QWEN36_KV_DTYPE','bf16'),
                              profile=os.environ.get('QWEN36_PROFILE','baseline64'),
                              context_limit=int(os.environ.get('QWEN36_CONTEXT','65536')),
                              moe_strategy=os.environ.get('QWEN36_STRATEGY','offload'),
                              memory_ratio=float(os.environ.get('QWEN36_MEMORY_RATIO','0.75')),
                              prefill_chunk=int(os.environ.get('QWEN36_PREFILL_CHUNK','2048')))
                try:
                    state = self.state_getter()
                    tr = state.stats
                    values.update(active_inference=tr.active, completed_requests=tr.completed,
                                  engine_prompt_tokens_total=tr.prompt_tokens_total,
                                  engine_completion_tokens_total=tr.completion_tokens_total,
                                  kv_used_pages=tr.kv_used_pages, kv_total_pages=tr.kv_total_pages,
                                  mamba_used_slots=tr.mamba_used_slots, mamba_total_slots=tr.mamba_total_slots,
                                  model_ready=getattr(state, 'ready_at', None) is not None,
                                  instance_id=getattr(state, 'instance_id', None))
                except (AssertionError, AttributeError, TypeError): pass
                self.writer.write('state.jsonl', values)

    async def aclose(self):
        async def finish_close():
            self.stop.set()
            if self.thread.is_alive(): await asyncio.to_thread(self.thread.join, 3)
            if not self.thread.is_alive():
                close = getattr(self.resources.source,'close',None)
                if close is not None: await asyncio.to_thread(close)
            await asyncio.to_thread(self.writer.close)
            self.closed = True
        if self.close_task is None: self.close_task = asyncio.create_task(finish_close())
        await asyncio.shield(self.close_task)

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            async def lifecycle_send(message):
                if message['type'] == 'lifespan.shutdown.complete': await self.aclose()
                await send(message)
            try: return await self.app(scope,receive,lifecycle_send)
            finally: await self.aclose()
        if scope['type'] != 'http': return await self.app(scope, receive, send)
        record = RequestRecord(scope, self.session_id, self.resources)
        scope['qwen36.metrics'] = record
        token = CURRENT.set(record)
        record.sample(self.resources.snapshot())
        if record.row['kind'] == 'generation':
            with self.lock: self.active_records.add(record)
        async def tracked_receive():
            message = await receive()
            if message['type'] == 'http.disconnect': record.disconnected = True
            return message
        async def tracked_send(message):
            if message['type'] == 'http.response.start':
                record.row['status'] = message['status']
                message = dict(message)
                message['headers'] = list(message.get('headers', [])) + [(b'x-qwen36-request-id', record.row['request_id'].encode())]
            await send(message)
        try:
            await self.app(scope, tracked_receive, tracked_send)
        except asyncio.CancelledError:
            record.row['outcome'] = 'cancelled'
            raise
        except Exception:
            record.row['outcome'] = 'application_error'
            raise
        finally:
            with self.lock:
                self.active_records.discard(record)
                row = record.finish(scope)
            self.writer.write('requests.jsonl', row)
            CURRENT.reset(token)


def install_native_observers(api_server, openai_api):
    """Read-only wrappers: no routing, weights, sampling or token stream changes."""
    original_ack = api_server.FrontendManager.wait_for_ack
    async def observed_ack(self, uid):
        global ENGINE_PROGRESS
        async with contextlib.aclosing(original_ack(self, uid)) as source:
            async for ack in source:
                ENGINE_PROGRESS=time.monotonic()
                record = CURRENT.get()
                if record is not None: record.acknowledge(ack)
                yield ack
    api_server.FrontendManager.wait_for_ack = observed_ack
    original_events = openai_api.generate_events
    async def observed_events(*args, **kwargs):
        async with contextlib.aclosing(original_events(*args, **kwargs)) as source:
            async for event in source:
                record = CURRENT.get()
                if record is not None: record.semantic(event)
                yield event
    openai_api.generate_events = observed_events
