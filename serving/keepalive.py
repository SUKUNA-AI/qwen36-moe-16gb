"""Bounded owner-generation lease for WSL. Stdlib only; no model or signals to others."""
import argparse
import datetime
import json
import os
import pathlib
import re
import signal
import stat
import sys
import time

STALE_SECONDS = 30.0
POLL_SECONDS = 1.0
# DrvFS can report ENOENT for a moment while Windows replaces the state file.
# Removal counts only after a second miss at least this long after the first.
MISSING_CONFIRM_SECONDS = 0.2
MAX_STATE_BYTES = 16384
RECORD_ROOT = pathlib.Path('/home/USER/qwen36-api/run/keepalive')


class HeartbeatLease:
    def __init__(self, generation, lease_id, owner_pid, owner_started, now):
        self.generation = generation
        self.lease_id = lease_id
        self.owner_pid = owner_pid
        self.owner_started = owner_started
        self.last_sequence = None
        self.last_progress = now

    def observe(self, state, now):
        if not isinstance(state, dict): return self.unreadable(now)
        if (state.get('Generation') != self.generation or
            type(state.get('SupervisorPid')) is not int or state['SupervisorPid'] != self.owner_pid or
            state.get('SupervisorStarted') != self.owner_started):
            return 'owner_changed'
        if state.get('LifetimeJobArmed') is not True: return 'ownership_not_armed'
        if state.get('KeepaliveLeaseId') != self.lease_id: return 'lease_replaced'
        sequence = state.get('HeartbeatSequence')
        if type(sequence) is not int or not 0 <= sequence < 2**63:
            return self.unreadable(now)
        if self.last_sequence is not None and sequence < self.last_sequence:
            return 'heartbeat_regressed'
        if self.last_sequence is None or sequence > self.last_sequence:
            self.last_sequence = sequence
            self.last_progress = now
        if now - self.last_progress >= STALE_SECONDS: return 'heartbeat_stale'
        return None

    def unreadable(self, now):
        return 'state_unreadable' if now - self.last_progress >= STALE_SECONDS else None


def read_owner_state(path):
    with path.open('rb') as handle: raw = handle.read(MAX_STATE_BYTES + 1)
    if len(raw) > MAX_STATE_BYTES: raise ValueError('state_size_limit')
    return json.loads(raw)


def private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError('record_directory_not_private')


def process_starttime():
    raw = pathlib.Path('/proc/self/stat').read_text()
    return int(raw[raw.rfind(')') + 2:].split()[19])


def write_record(path, record, create=False):
    data = (json.dumps(record, sort_keys=True) + '\n').encode()
    target = path if create else path.with_name(path.name + '.new')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, 'wb') as handle: handle.write(data)
    if not create: target.replace(path)


def run(state_path, generation, lease_id, owner_pid, owner_started, record_root=RECORD_ROOT):
    if sys.platform != 'linux' or os.getuid() == 0: raise ValueError('ordinary_linux_owner_required')
    if not re.fullmatch(r'[a-f0-9]{32}', generation): raise ValueError('invalid_generation')
    if not re.fullmatch(r'[a-f0-9]{32}', lease_id): raise ValueError('invalid_lease_id')
    if type(owner_pid) is not int or owner_pid < 1 or not owner_started: raise ValueError('invalid_owner')
    os.umask(0o077)
    private_directory(record_root)
    generation_dir = record_root / generation
    private_directory(generation_dir)
    linux_pid = os.getpid(); starttime = process_starttime()
    record_path = generation_dir / f'{linux_pid}-{starttime}.json'
    record = {'schema':1,'generation':generation,'lease_id':lease_id,'windows_owner_pid':owner_pid,
        'windows_owner_started':owner_started,'linux_pid':linux_pid,
        'linux_starttime_ticks':starttime,'linux_uid':os.getuid(),
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'state':'running','stale_seconds':STALE_SECONDS}
    write_record(record_path, record, create=True)
    lease = HeartbeatLease(generation, lease_id, owner_pid, owner_started, time.monotonic())
    interrupted = False
    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
    handlers = {sig:signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    reason = 'internal_error'
    missing_since = None
    try:
        while True:
            if interrupted:
                reason = 'interrupted'; break
            now = time.monotonic()
            try:
                current = read_owner_state(state_path)
            except FileNotFoundError:
                if missing_since is None:
                    missing_since = now
                elif now - missing_since >= MISSING_CONFIRM_SECONDS:
                    reason = 'owner_state_removed'; break
                # Re-check soon; a miss never refreshes the heartbeat lease.
                outcome = lease.unreadable(now)
                if outcome:
                    reason = outcome; break
                time.sleep(MISSING_CONFIRM_SECONDS)
                continue
            except (OSError, ValueError, RecursionError, UnicodeError):
                missing_since = None
                outcome = lease.unreadable(now)
            else:
                missing_since = None
                outcome = lease.observe(current, now)
            if outcome:
                reason = outcome; break
            time.sleep(POLL_SECONDS)
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        record.update(state='exited',reason=reason,last_sequence=lease.last_sequence,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_record(record_path, record)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=pathlib.Path, required=True)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--lease-id', required=True)
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--owner-started', required=True)
    parser.add_argument('--record-root', type=pathlib.Path, default=RECORD_ROOT,
        help='Owner-private directory; override only for isolated synthetic tests')
    args = parser.parse_args()
    try:
        return run(args.state, args.generation, args.lease_id, args.owner_pid, args.owner_started, args.record_root)
    except Exception as error:
        # No state contents, filesystem path, command line or secrets in stderr.
        print(json.dumps({'keepalive':'failed','error_type':type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
