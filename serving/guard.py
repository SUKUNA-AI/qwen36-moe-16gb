"""Owner-controlled admission/authentication for the pinned FreeToken 0.1.3 app.
The runtime package is not edited. All HTTP routes require the persistent owner key
or a per-user key from secrets/users.env (KeyRing).
Only the two public API routes and authenticated local diagnostics are admitted.
The engine runs one generation at a time; QWEN36_QUEUE_DEPTH (default 0: reject
with 429) lets that many further requests wait in FIFO order instead.
"""
import asyncio, collections, contextlib, hmac, json, math, os, re, stat, time

MAX_BODY = 2 * 1024 * 1024
MAX_SECONDS = 900
BODY_SECONDS = 15
CLEANUP_SECONDS = 2
PROTOCOL_FAILURE_AT = None
_ABORT_TASKS = {}
CONTEXT = int(os.environ.get('QWEN36_CONTEXT', '65536'))
# Ingress admits 8 HTTP requests: one generation plus at most 6 waiting leaves
# a connection for diagnostics. Queue wait counts toward MAX_SECONDS (the edge
# allows 910 s), so at least 60 s of generation always remains.
QUEUE_DEPTH_MAX = 6
QUEUE_WAIT_MAX = MAX_SECONDS - 60
USER_KEYS_MAX = 64
USER_NAME = re.compile(r'[a-z0-9][a-z0-9_-]{0,31}')
USER_KEY = re.compile(r'[A-Za-z0-9_-]{43,128}')
ALLOWED = {
    ('GET','/v1/models'), ('POST','/v1/chat/completions'),
    ('GET','/health'), ('GET','/v1/stats'), ('GET','/v1/cache/status'),
}

def bounded_setting(name, default, low, high, environ=None):
    """Owner serving setting (config/runtime.env); absent or empty means the default."""
    raw = (os.environ if environ is None else environ).get(name, '')
    if raw == '': return default
    if not (raw.isascii() and raw.isdigit() and len(raw) <= 4 and low <= int(raw) <= high):
        raise RuntimeError(f'{name} must be an integer in {low}..{high}')
    return int(raw)

def queue_settings(environ=None):
    """(depth, wait seconds) for the admission queue; serve_guarded checks them before model load."""
    return (bounded_setting('QWEN36_QUEUE_DEPTH', 0, 0, QUEUE_DEPTH_MAX, environ),
            bounded_setting('QWEN36_QUEUE_WAIT', 600, 1, QUEUE_WAIT_MAX, environ))

def parse_user_keys(text, owner):
    """name=key lines; blank and # lines ignored. Names and keys are unique, owner is reserved."""
    users, keys = {}, {owner}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        name, sep, key = line.partition('=')
        if not sep or not USER_NAME.fullmatch(name) or name == 'owner' or name in users:
            raise ValueError('Invalid user key entry')
        if not USER_KEY.fullmatch(key) or key.encode('ascii') in keys:
            raise ValueError('Invalid user key entry')
        users[name] = key.encode('ascii')
        keys.add(users[name])
    if len(users) > USER_KEYS_MAX: raise ValueError('Too many user keys')
    return users

class KeyRing:
    """The unit's owner key plus optional per-user keys.

    The file is re-read whenever its identity changes, so keys are issued or
    revoked without a restart. A symlink, group/other access or any bad line
    fails closed to the owner key alone.
    """
    def __init__(self, owner, path=None):
        self.owner = owner
        self.path = path
        self.identity = None
        self.users = {}
        self.invalid = False

    def refresh(self):
        if self.path is None: return
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            self.identity, self.users, self.invalid = None, {}, False
            return
        except OSError:
            info = None
        identity = None if info is None else (info.st_ino, info.st_mtime_ns, info.st_size, info.st_mode)
        if identity is not None and identity == self.identity: return
        self.identity = identity
        try:
            if info is None: raise ValueError('Unreadable user key file')
            descriptor = os.open(self.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(descriptor, 'rb') as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_mode & 0o077:
                    raise ValueError('Unsafe user key file')
                data = handle.read(65537)
            if len(data) > 65536: raise ValueError('User key file too large')
            # A Windows editor may prepend a UTF-8 BOM; nothing else non-ASCII is accepted.
            data = data.removeprefix(b'\xef\xbb\xbf')
            self.users, self.invalid = parse_user_keys(data.decode('ascii'), self.owner), False
        except (OSError, ValueError, UnicodeError):
            self.users = {}
            if not self.invalid: print('QWEN36_USER_KEYS_INVALID', flush=True)
            self.invalid = True

    def lookup(self, header):
        """Client name whose Bearer key matches, else None. Every key is compared."""
        self.refresh()
        found = 'owner' if hmac.compare_digest(header, b'Bearer ' + self.owner) else None
        for name, key in self.users.items():
            if hmac.compare_digest(header, b'Bearer ' + key) and found is None: found = name
        return found

_KEYRING = None

def keyring():
    """Process-wide KeyRing shared by Ingress and APIGuard."""
    global _KEYRING
    owner = os.environ['QWEN36_API_KEY'].encode('ascii')
    if len(owner) < 43: raise RuntimeError('API key has insufficient encoded length')
    if _KEYRING is None or _KEYRING.owner != owner:
        root = os.environ.get('QWEN36_DATA_ROOT')
        _KEYRING = KeyRing(owner, os.path.join(root, 'secrets', 'users.env') if root else None)
    return _KEYRING

def validate_json_depth(body, maximum=64):
    depth, quoted, escaped = 0, False, False
    for byte in body:
        if quoted:
            if escaped: escaped = False
            elif byte == 92: escaped = True
            elif byte == 34: quoted = False
        elif byte == 34: quoted = True
        elif byte in (91,123):
            depth += 1
            if depth > maximum: raise ValueError('JSON nesting exceeds 64 levels')
        elif byte in (93,125): depth -= 1

def invalid_constant(value):
    raise ValueError('Non-finite JSON number')

def finite_float(value):
    number=float(value)
    if not math.isfinite(number):raise ValueError('Non-finite JSON number')
    return number

def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result

def start_abort(state, uid):
    """Retain and deduplicate abort tasks even when a request task is cancelled."""
    key = (id(state), uid)
    task = _ABORT_TASKS.get(key)
    if task is not None: return task
    task = asyncio.create_task(state.abort_user(uid))
    _ABORT_TASKS[key] = task
    def completed(future):
        _ABORT_TASKS.pop(key, None)
        if not future.cancelled() and future.exception() is not None:
            print('QWEN36_ABORT_DISPATCH_FAILED', flush=True)
    task.add_done_callback(completed)
    return task

def validate_payload(payload):
    if not isinstance(payload,dict): raise ValueError('JSON object required')
    if payload.get('model') != 'qwen36-35b': raise ValueError('Unknown model id')
    if not isinstance(payload.get('messages'),list) or not payload['messages']:
        raise ValueError('A non-empty messages array is required')
    # The pinned engine has no constrained decoder for required/named calls.
    # Reject before admission or SSE; parser recognition cannot force a call.
    tool_choice = payload.get('tool_choice')
    if tool_choice is not None and (type(tool_choice) is not str or tool_choice not in ('auto', 'none')):
        raise ValueError('tool_choice supports only auto or none; forced calls are unavailable')
    # GenSpec drops parallel_tool_calls (generation emits every parsed call) and
    # the schema drops strict; legacy functions never reach the template. Reject
    # instead of silently accepting guarantees that are not enforced.
    tools, parallel = payload.get('tools'), payload.get('parallel_tool_calls')
    if parallel is not None and type(parallel) is not bool:
        raise ValueError('parallel_tool_calls must be a boolean')
    if parallel is False and tools and tool_choice != 'none':
        raise ValueError('parallel_tool_calls=false is unsupported; a single tool call is not enforced')
    for tool in tools if isinstance(tools, list) else ():
        for owner in (tool, tool.get('function')) if isinstance(tool, dict) else ():
            if isinstance(owner, dict) and owner.get('strict') is not None and owner.get('strict') is not False:
                raise ValueError('strict function schemas are unsupported; omit strict or set it to false')
    if payload.get('functions') is not None:
        raise ValueError('Legacy functions are unsupported; use tools with tool_choice auto or none')
    if type(payload.get('n',1)) is not int or payload.get('n',1) != 1:
        raise ValueError('Only n=1 is supported')
    for field in ('max_tokens','max_completion_tokens'):
        if field in payload and (type(payload[field]) is not int or not 1 <= payload[field] <= 8192):
            raise ValueError('Output budget must be 1..8192 tokens')
    if 'max_tokens' in payload and 'max_completion_tokens' in payload and payload['max_tokens'] != payload['max_completion_tokens']:
        raise ValueError('Conflicting output budgets')
    for field, low, high in (('temperature',0,2),('top_p',0,1)):
        value = payload.get(field)
        # Check bounds before isfinite: converting a very large JSON integer to
        # float raises OverflowError and used to turn a client error into 500.
        if value is not None and (type(value) not in (int,float) or not low <= value <= high or not math.isfinite(value) or field == 'top_p' and value == 0):
            raise ValueError('Invalid sampling parameter')
    if 'top_k' in payload and (type(payload['top_k']) is not int or payload['top_k'] not in (-1,0) and not 1 <= payload['top_k'] <= 248320):
        raise ValueError('Invalid top_k')
    if 'seed' in payload and (type(payload['seed']) is not int or not 0 <= payload['seed'] < 2**63):
        raise ValueError('seed must be an integer in 0..2^63-1')
    for field in ('presence_penalty','frequency_penalty'):
        value=payload.get(field,0)
        if type(value) not in (int,float) or value != 0: raise ValueError('Only numeric zero penalties are supported')
    for field in ('stream','ignore_eos'):
        if field in payload and type(payload[field]) is not bool:
            raise ValueError('stream and ignore_eos must be booleans')
    options = payload.get('stream_options')
    if options is not None:
        if not isinstance(options, dict) or ('include_usage' in options and type(options['include_usage']) is not bool):
            raise ValueError('Invalid stream_options')
    if payload.get('logprobs') not in (None, False) or payload.get('top_logprobs') not in (None, 0):
        raise ValueError('logprobs is not supported')
    if payload.get('audio') is not None or payload.get('modalities') not in (None, ['text']):
        raise ValueError('Only text input and output are supported')
    for message in payload['messages']:
        if not isinstance(message,dict): raise ValueError('Each message must be an object')
        content = message.get('content')
        if isinstance(content,list):
            if any(not isinstance(p,dict) or p.get('type') != 'text' or not isinstance(p.get('text'),str) for p in content):
                raise ValueError('Only text content parts are supported')
        elif content is not None and not isinstance(content,str):
            raise ValueError('Message content must be text')

class APIGuard:
    def __init__(self, app, state_getter=None, queue_depth=None, queue_wait=None):
        self.app = app
        self.keys = keyring()
        self.state_getter = state_getter
        depth, wait = queue_settings()
        self.queue_depth = depth if queue_depth is None else queue_depth
        self.queue_wait = wait if queue_wait is None else queue_wait
        if not (0 <= self.queue_depth <= QUEUE_DEPTH_MAX and 0 < self.queue_wait <= QUEUE_WAIT_MAX):
            raise RuntimeError('Invalid queue settings')
        self.waiters = collections.deque()
        self.busy = False
        self.cleanup_failed = False
        self.bucket = 60.0
        self.refilled = time.monotonic()

    async def reject(self, send, status, message, retry=False):
        headers=[(b'content-type',b'application/json'),(b'cache-control',b'no-store')]
        if retry: headers.append((b'retry-after',b'2'))
        await send({'type':'http.response.start','status':status,'headers':headers})
        await send({'type':'http.response.body','body':json.dumps({'error':{'message':message}}).encode()})

    def active(self):
        if not self.state_getter: return 0
        try: return self.state_getter().stats.active
        except (AssertionError,AttributeError): return 0

    def dispatch(self):
        """Hand the single generation slot to the oldest waiter (no await inside)."""
        if self.cleanup_failed:
            while self.waiters:
                waiter=self.waiters.popleft()
                if not waiter.done(): waiter.set_result(False)
            return
        if self.busy or not self.waiters or self.active(): return
        self.busy=True
        self.waiters.popleft().set_result(True)

    async def wait_turn(self, receive, deadline):
        """FIFO wait for the slot. Returns (outcome, pending disconnect probe)."""
        waiter=asyncio.get_running_loop().create_future()
        self.waiters.append(waiter)
        probe=asyncio.create_task(receive())
        outcome=None
        try:
            while outcome is None:
                # Polling also covers a slot freed while the engine was still
                # acknowledging an abort, when no request is left to dispatch.
                self.dispatch()
                if probe.done():
                    if probe.cancelled() or probe.exception() is not None or probe.result().get('type')=='http.disconnect':
                        outcome='disconnected'
                        break
                    probe=asyncio.create_task(receive())
                if waiter.done():
                    outcome='admitted' if waiter.result() else 'failed'
                    break
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    outcome='timeout'
                    break
                await asyncio.wait({waiter,probe},timeout=min(.25,remaining),return_when=asyncio.FIRST_COMPLETED)
        finally:
            if waiter in self.waiters: self.waiters.remove(waiter)
            if outcome!='admitted':
                if waiter.done() and not waiter.cancelled() and waiter.result(): self.busy=False
                elif not waiter.done(): waiter.cancel()
                if not probe.done(): probe.cancel()
                self.dispatch()
        return outcome, probe

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'websocket': return await send({'type':'websocket.close','code':1008})
        if scope['type'] != 'http': return await self.app(scope, receive, send)
        route=(scope['method'],scope['path'])
        if route not in ALLOWED: return await self.reject(send,404,'Route unavailable')
        auth=[v for k,v in scope['headers'] if k.lower()==b'authorization']
        if len(auth)!=1 or self.keys.lookup(auth[0]) is None:
            return await self.reject(send,401,'Invalid or missing Bearer token')
        now=time.monotonic()
        self.bucket=min(60.0,self.bucket+(now-self.refilled))
        self.refilled=now
        if self.bucket<1: return await self.reject(send,429,'Request rate exceeded',True)
        self.bucket-=1
        generation=route==('POST','/v1/chat/completions')
        if self.cleanup_failed and (generation or route==('GET','/health')):
            return await self.reject(send,503,'Protocol cleanup failed; service recovery pending',True)
        if generation and (self.busy or self.active()) and len(self.waiters)>=self.queue_depth:
            if not self.queue_depth:
                return await self.reject(send,429,'One inference request is already active; no waiting queue',True)
            return await self.reject(send,429,'Inference queue is full',True)
        headers=dict(scope['headers'])
        if sum(k.lower()==b'content-length' for k,v in scope['headers']) > 1:
            return await self.reject(send,400,'Duplicate content length')
        try:
            content_length=int(headers.get(b'content-length',b'0'))
        except ValueError:
            return await self.reject(send,400,'Invalid content length')
        if content_length < 0: return await self.reject(send,400,'Invalid content length')
        if content_length>MAX_BODY: return await self.reject(send,413,'HTTP body exceeds 2 MiB')
        body=bytearray()
        probe=None
        budget=MAX_SECONDS
        try:
            async with asyncio.timeout(BODY_SECONDS):
                while True:
                    message=await receive()
                    if message['type']=='http.disconnect': return
                    body.extend(message.get('body',b''))
                    if len(body)>MAX_BODY: return await self.reject(send,413,'HTTP body exceeds 2 MiB')
                    if not message.get('more_body',False): break
        except TimeoutError:
            return await self.reject(send,408,'HTTP body read timed out')
        if generation:
            try:
                validate_json_depth(body)
                payload=json.loads(body,parse_constant=invalid_constant,parse_float=finite_float,
                                   object_pairs_hook=no_duplicate_keys)
                validate_payload(payload)
            except (ValueError,UnicodeError,RecursionError) as exc:
                message = str(exc) if type(exc) is ValueError else 'Invalid JSON'
                return await self.reject(send,400,message)
            metrics=scope.get('qwen36.metrics')
            if metrics is not None: metrics.configure(payload)
            # No await between the admission check and this assignment.
            if self.busy or self.active() or self.waiters:
                if len(self.waiters)>=self.queue_depth:
                    if not self.queue_depth:
                        return await self.reject(send,429,'Inference is busy; no waiting queue',True)
                    return await self.reject(send,429,'Inference queue is full',True)
                queued=time.monotonic()
                outcome,probe=await self.wait_turn(receive,queued+self.queue_wait)
                if outcome=='disconnected': return
                if outcome=='failed':
                    return await self.reject(send,503,'Protocol cleanup failed; service recovery pending',True)
                if outcome=='timeout':
                    return await self.reject(send,429,'Inference queue wait exceeded',True)
                waited=time.monotonic()-queued
                # The whole request, queue included, stays inside MAX_SECONDS.
                budget=max(1.0,MAX_SECONDS-waited)
                if metrics is not None: metrics.row['queue_wait_s']=waited
            else:
                self.busy=True
        first=True
        disconnected=asyncio.Event()
        started=False

        async def replay_receive():
            nonlocal first
            if first:
                first=False
                return {'type':'http.request','body':bytes(body),'more_body':False}
            await disconnected.wait()
            return {'type':'http.disconnect'}

        async def watch_disconnect():
            nonlocal probe
            while True:
                if probe is not None:
                    pending,probe=probe,None
                    m=await pending
                else: m=await receive()
                if m['type']=='http.disconnect':
                    disconnected.set()
                    return

        async def tracked_send(message):
            nonlocal started
            if message['type']=='http.response.start': started=True
            await send(message)

        app_task=asyncio.create_task(self.app(scope,replay_receive,tracked_send))
        watcher=asyncio.create_task(watch_disconnect())
        normal_completion=False
        try:
            done,_=await asyncio.wait({app_task,watcher},timeout=budget,return_when=asyncio.FIRST_COMPLETED)
            if app_task in done:
                await app_task
                normal_completion=True
            else:
                # Send abort before cancelling the protocol consumer: nonstream
                # cancellation used to unwind its waiter before dispatching abort.
                record=scope.get('qwen36.metrics')
                uid=getattr(record,'engine_uid',None)
                if generation and uid is not None and self.state_getter:
                    state=self.state_getter()
                    if uid in state.stats.inflight_uids:
                        task=start_abort(state,uid)
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(asyncio.shield(task),CLEANUP_SECONDS)
                app_task.cancel()
                if not done: scope['qwen36.outcome']='timeout'
                if not done and not started:
                    await self.reject(send,504,'Maximum request duration exceeded')
        finally:
            async def cleanup():
                global PROTOCOL_FAILURE_AT
                if generation and not normal_completion and self.state_getter:
                    record=scope.get('qwen36.metrics')
                    uid=getattr(record,'engine_uid',None)
                    if uid is not None:
                        state=self.state_getter()
                        if uid in state.stats.inflight_uids: start_abort(state,uid)
                for task in (watcher,app_task):
                    if not task.done(): task.cancel()
                _, pending = await asyncio.wait({watcher,app_task},timeout=CLEANUP_SECONDS)
                for task in (watcher,app_task):
                    if task.done() and not task.cancelled(): task.exception()
                if pending:
                    self.cleanup_failed=True
                    if PROTOCOL_FAILURE_AT is None: PROTOCOL_FAILURE_AT=time.monotonic()
                    for task in pending:
                        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                if generation:
                    # Native stats retain aborted uids until the scheduler's
                    # terminal reply; this is a worker acknowledgement barrier.
                    deadline=time.monotonic()+5
                    while self.active() and time.monotonic()<deadline: await asyncio.sleep(.05)
                    self.busy=self.cleanup_failed
                    self.dispatch()
            task=asyncio.create_task(cleanup())
            # A second server cancellation must not abandon the cleanup task.
            try: await asyncio.shield(task)
            except asyncio.CancelledError:
                task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                raise
