"""Per-prefill H2D / compute overlap from an nsys sqlite export (S4 C0 evidence).

Each host NVTX range 'Qwen36/prefill' owns the GPU work whose runtime call it enclosed (CUPTI
correlation ids). Per prefill: GPU window, kernel and H2D busy time, their overlap, exposed H2D,
H2D bursts (H2D ops merged across gaps < 0.3 ms) with kernel occupancy inside bursts and in the
gaps between them, and the launch -> start delay of kernels on the copy stream(s) -- the C0
mechanism (bookkeeping kernels queued behind the expert GEMM).

Usage: nsys_h2d.py REPORT.sqlite [--labels a,b,...] [--out FILE.json]
"""
import argparse
import json
import sqlite3
import statistics

GAP_NS = 300_000


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def inter(a, b):
    i = j = 0
    t = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def clip(iv, s, e):
    return [[max(a, s), min(b, e)] for a, b in iv if b > s and a < e]


def med(xs):
    return statistics.median(xs) if xs else None


def tables(conn):
    return {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}


def nvtx_ranges(conn, names):
    strings = {}
    if 'StringIds' in names:
        strings = dict(conn.execute('select id, value from StringIds'))
    rows = conn.execute('select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null').fetchall()
    return [(s, e, tid) for s, e, text, text_id, tid in rows
            if (text or strings.get(text_id, '')) == 'Qwen36/prefill']


def analyze(path, labels):
    conn = sqlite3.connect(path)
    names = tables(conn)
    ranges = sorted(nvtx_ranges(conn, names))
    runtime = conn.execute('select start, end, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME').fetchall()
    kernels = conn.execute('select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    memcpy = conn.execute('select start, end, streamId, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
    rt_by_corr = {c: (s, e, tid) for s, e, c, tid in runtime}
    out = []
    for idx, (rs, re_, tid) in enumerate(ranges):
        corr = {c for c, (s, e, t) in rt_by_corr.items() if t == tid and rs <= s <= re_}
        ks = [k for k in kernels if k[3] in corr]
        ms = [m for m in memcpy if m[3] in corr]
        if not ks:
            continue
        ops = [(k[0], k[1]) for k in ks] + [(m[0], m[1]) for m in ms]
        w0, w1 = min(s for s, _ in ops), max(e for _, e in ops)
        kiv = merge([(k[0], k[1]) for k in ks])
        h2d = [m for m in ms if m[5] == 1]
        hiv = merge([(m[0], m[1]) for m in h2d])
        h2d_bytes = sum(m[4] for m in h2d)
        d2d_bytes = sum(m[4] for m in ms if m[5] == 8)
        by_stream = {}
        for m in h2d:
            by_stream[m[2]] = by_stream.get(m[2], 0) + m[4]
        copy_stream = max(by_stream, key=by_stream.get) if by_stream else None
        ck = sorted(k for k in ks if k[2] == copy_stream)
        # queueing delay of a copy-stream kernel: start minus the later of its launch call and the
        # end of the previous op on the copy stream (time it waited for SMs, not for its inputs)
        prev_end = {}
        stream_ops = sorted([(k[0], k[1], 'k', k[3]) for k in ck] +
                            [(m[0], m[1], 'm', m[3]) for m in ms if m[2] == copy_stream])
        delays, last = [], None
        for s0, e0, kind, c in stream_ops:
            if kind == 'k' and c in rt_by_corr:
                ready = max(rt_by_corr[c][1], last if last is not None else 0)
                delays.append(max(0, s0 - ready) / 1e6)
            last = e0 if last is None else max(last, e0)
        bursts = []
        for s, e in hiv:
            if bursts and s - bursts[-1][1] < GAP_NS:
                bursts[-1][1] = e
            else:
                bursts.append([s, e])
        bdetail, gdetail = [], []
        for i, (s, e) in enumerate(bursts):
            b_bytes = sum(m[4] for m in h2d if m[0] >= s and m[1] <= e)
            bdetail.append({'ms': (e - s) / 1e6, 'gbps': b_bytes / max(1, e - s),
                            'kernel_busy_pct': 100 * total(clip(kiv, s, e)) / max(1, e - s)})
            if i + 1 < len(bursts):
                g0, g1 = e, bursts[i + 1][0]
                gdetail.append({'ms': (g1 - g0) / 1e6, 'kernel_busy_pct': 100 * total(clip(kiv, g0, g1)) / max(1, g1 - g0)})
        h_ms, k_ms, ov = total(hiv) / 1e6, total(kiv) / 1e6, inter(hiv, kiv) / 1e6
        out.append({
            'index': idx, 'label': labels[len(out)] if len(out) < len(labels) else None,
            'host_range_ms': (re_ - rs) / 1e6, 'gpu_window_ms': (w1 - w0) / 1e6,
            'kernel_busy_ms': k_ms, 'h2d_busy_ms': h_ms, 'h2d_kernel_overlap_ms': ov,
            'h2d_overlap_pct': 100 * ov / h_ms if h_ms else None, 'exposed_h2d_ms': h_ms - ov,
            'h2d_gb': h2d_bytes / 1e9, 'h2d_gbps_while_copying': h2d_bytes / max(1, total(hiv)),
            'd2d_gb': d2d_bytes / 1e9, 'copy_stream': copy_stream, 'copy_stream_kernels': len(ck),
            'copy_stream_kernel_ms': sum(k[1] - k[0] for k in ck) / 1e6,
            'copy_stream_kernel_queue_ms': {'median': med(delays), 'max': max(delays) if delays else None,
                                               'sum': sum(delays) if delays else 0.0},
            'bursts': len(bursts), 'median_burst_ms': med([b['ms'] for b in bdetail]),
            'median_burst_gbps': med([b['gbps'] for b in bdetail]),
            'median_kernel_busy_in_burst_pct': med([b['kernel_busy_pct'] for b in bdetail]),
            'median_gap_ms': med([g['ms'] for g in gdetail]),
            'median_kernel_busy_in_gap_pct': med([g['kernel_busy_pct'] for g in gdetail]),
            'sum_gap_ms': sum(g['ms'] for g in gdetail),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sqlite')
    p.add_argument('--labels', default='')
    p.add_argument('--out')
    a = p.parse_args()
    res = analyze(a.sqlite, [x for x in a.labels.split(',') if x])
    text = json.dumps(res, indent=1)
    if a.out:
        open(a.out, 'w').write(text)
    for r in res:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
               if k not in ('copy_stream_kernel_queue_ms',)}, r['copy_stream_kernel_queue_ms'])


if __name__ == '__main__':
    main()
