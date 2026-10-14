"""Bound unauthenticated work before body parsing and numeric telemetry."""
import hmac
import os
import time
from guard import APIGuard, ALLOWED

COUNTERS = {'rate_rejected':0, 'connections_rejected':0, 'unauthorized':0}

class Ingress:
    def __init__(self, app):
        self.app = app
        self.active = 0
        self.bucket = 20.0
        self.refilled = time.monotonic()
        self.expected = b'Bearer ' + os.environ['QWEN36_API_KEY'].encode('ascii')

    async def reject(self, send, code, message):
        await APIGuard.reject(self, send, code, message, code in (429,503))

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'websocket':
            return await send({'type':'websocket.close', 'code':1008})
        if scope['type'] != 'http': return await self.app(scope,receive,send)
        now = time.monotonic()
        self.bucket = min(20.0, self.bucket + (now-self.refilled)*10)
        self.refilled = now
        if self.bucket < 1:
            COUNTERS['rate_rejected'] += 1
            return await self.reject(send,429,'Request rate exceeded')
        self.bucket -= 1
        if (scope['method'],scope['path']) not in ALLOWED:
            return await self.reject(send,404,'Route unavailable')
        auth = [v for k,v in scope['headers'] if k.lower() == b'authorization']
        if len(auth) != 1 or not hmac.compare_digest(auth[0],self.expected):
            COUNTERS['unauthorized'] += 1
            return await self.reject(send,401,'Invalid or missing Bearer token')
        if self.active >= 8:
            COUNTERS['connections_rejected'] += 1
            return await self.reject(send,503,'HTTP concurrency limit reached')
        self.active += 1
        try: await self.app(scope,receive,send)
        finally: self.active -= 1
