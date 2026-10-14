"""Parent fairness wrapper (same idea as work/agents/parent/parent-screen.py) for the frozen native tools that take the
GPU-exclusive lock themselves (matrix.py, tool-pilot.py, compare-ab.py). The CALLER holds the real GPU lock with a
blocking `flock -w 3600 -E 75 LOCK ...` around this process; the tool's own non-blocking lock is pointed at the
private dummy lock file parent-screen.py already uses, so the parent queues as a blocking waiter (fair access to the
GPU) without modifying any frozen harness file. Exclusivity is unchanged: the real lock is held for the whole call.

Fail-closed: refuses (exit 75, nothing started) unless the real lock is held by an ANCESTOR process (the flock(1)
wrapper), so it can never run a matrix without the real GPU lock.

Usage: parent-lockwrap.py {matrix|tool-pilot|compare-ab} ARGS...   (exit code = the tool's own exit code)
"""
import importlib.util
import os
import sys

N = '/home/USER/qwen36-api/optimization'
LOCK = '/home/USER/qwen36-api/run/gpu-exclusive.lock'
DUMMY = '/home/USER/qwen36-api/run/parent-screen-dummy.lock'
TOOLS = {'matrix': 'matrix.py', 'tool-pilot': 'tool-pilot.py', 'compare-ab': 'compare-ab.py'}


def ancestors():
    pid, seen = os.getpid(), []
    while pid and pid not in seen:
        seen.append(pid)
        try:
            with open(f'/proc/{pid}/status', encoding='utf-8') as handle:
                pid = next((int(line.split()[1]) for line in handle if line.startswith('PPid:')), 0)
        except OSError:
            break
    return set(seen)


def real_lock_held_by_ancestor():
    try:
        ino = os.stat(LOCK).st_ino
        with open('/proc/locks', encoding='utf-8') as handle:
            rows = [line.split() for line in handle]
    except OSError:
        return False
    holders = set()
    for row in rows:
        # "1: FLOCK  ADVISORY  WRITE 12345 00:2f:123456 0 EOF" (waiters carry "->" and are skipped)
        if len(row) >= 6 and row[1] == 'FLOCK' and row[2] != '->':
            try:
                if int(row[5].split(':')[-1]) == ino:
                    holders.add(int(row[4]))
            except ValueError:
                continue
    return bool(holders & ancestors())


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print('usage: parent-lockwrap.py {matrix|tool-pilot|compare-ab} ARGS...', file=sys.stderr)
        return 2
    if not real_lock_held_by_ancestor():
        print('{"result":"refused","reason":"real GPU lock not held by an ancestor flock wrapper"}', file=sys.stderr)
        return 75
    sys.path.insert(0, N)
    name = sys.argv[1]
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), os.path.join(N, TOOLS[name]))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(sys.argv[2:], lock_path=DUMMY)


if __name__ == '__main__':
    raise SystemExit(main())
