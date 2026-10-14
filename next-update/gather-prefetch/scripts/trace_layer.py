"""Print the kernel sequence of one decode step (warm decode) from the f1 nsys sqlite."""
import sqlite3, sys, bisect, collections
db = sys.argv[1]
which = int(sys.argv[2]) if len(sys.argv) > 2 else 200
con = sqlite3.connect(db)
S = dict(con.execute("select id, value from StringIds"))
nv = [(s, e, (t if t is not None else S.get(tid, ""))) for s, e, t, tid in
      con.execute("select start, end, text, textId from NVTX_EVENTS where end is not null")]
dec = sorted((s, e) for s, e, t in nv if t == "Qwen36/decode")
print("decode ranges", len(dec))
api = {c: s for c, s in con.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
ks = con.execute("select start, end, correlationId, shortName, demangledName, streamId, gridX, blockX from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
# graph launches: kernels from graph replay share correlation id of cudaGraphLaunch
dstarts = [s for s, e in dec]
per = collections.defaultdict(list)
for s, e, c, sn, dn, st, gx, bx in ks:
    t = api.get(c)
    if t is None: continue
    i = bisect.bisect_right(dstarts, t) - 1
    if i >= 0 and dec[i][0] <= t <= dec[i][1]:
        per[i].append((s, e, S.get(sn, ""), st, gx, bx))
steps = sorted(per)
i = steps[min(which, len(steps)-1)]
kl = sorted(per[i])
t0 = kl[0][0]
for s, e, n, st, gx, bx in kl[:int(sys.argv[3]) if len(sys.argv) > 3 else 80]:
    print(f"{(s-t0)/1e3:9.2f} {(e-s)/1e3:8.2f} st{st} g{gx}x{bx} {n[:90]}")
