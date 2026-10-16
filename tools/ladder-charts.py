#!/usr/bin/env python3
"""Charts and tables for the cumulative main128 release ladder (L0 f1 .. L8 candidate).

No third-party packages; the SVG style, themes and helpers come from
tools/readme_charts.py, so the ladder charts look like the README charts.

    python3 tools/ladder-charts.py                      # results/ladder/summary.json -> results/ladder/charts/
    python3 tools/ladder-charts.py --input harness/fixtures/ladder-summary-synthetic.json   # -> charts-draft/

Input: schema "qwen36-ladder-summary" v1 (written by the GPU ladder session).
Every field except schema, schema_version and steps[].id may be missing or null;
the charts then print "н/д", so the file can be filled step by step.

- lengths: prompt token counts; per-length maps use them as string keys.
- steps[]: id, label, lever, config, parent, cumulative (false for side arms such
  as T3only), teacher {status pass|fail|pending|not_run, passed, total, max_abs},
  ab {reference, pairs, ttft_s, ref_ttft_s, decode_tps, ref_decode_tps}: raw
  per-pair values, the arm and its reference in the same round,
  memory {memory_ratio, expert_slots, process_peak_mib, free_vram_min_mib,
  windows_shared_drift_mib, misses_per_token {length or "all": n}},
  nsys {decode_16k {ms_per_step, buckets_ms}, prefill_short / prefill_16k
  {window_ms, buckets_ms, h2d_busy_ms, h2d_overlap_pct}}.
  buckets_ms keys are the buckets_requested keys of results/nsight/dashboard.json.
- release_ab: {candidate, reference, pairs, ttft_s, ref_ttft_s, decode_tps,
  ref_decode_tps, verdict}.
- synthetic: true marks a fixture. Synthetic input is never written to
  results/ladder/charts; every chart then carries a "draft" badge.

A lever's contribution is the median of its per-pair ratios against its parent
(arm / reference in the same round), chained from the L0 median. When the pairs
are missing or ran against another arm, the ratio of medians is used and the
chart says so.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readme_charts as rc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = os.path.join(ROOT, "results", "ladder")
SCHEMA = "qwen36-ladder-summary"
SCHEMA_VERSION = 1
WINDOWS_RESERVE_MIB = 1024  # owner rule: 1 GiB of VRAM for Windows, the rest for the model

THEMES = rc.THEMES
esc = rc.esc

# Axis captions; a step's own "label" wins in tables.
SHORT = {
    "L0": "f1", "L1": "+T1", "L2": "+T2", "L3": "+T3", "L4": "+эмб. на хосте",
    "L5": "+ratio 0.85", "L6": "+ядра S3", "L7": "+C0", "L8": "+C1", "T3only": "T3 отдельно",
}

# dashboard.json buckets_requested keys -> theme colour, legend label
CATS = [
    ("moe_experts", "moe", "Эксперты MoE"),
    ("fp8_gemv_gemm", "fp8", "Плотные FP8 и o-proj"),
    ("decode_gather", "gather", "Подкачка экспертов (gather)"),
    ("full_attention", "attn", "Attention"),
    ("gdn", "gdn", "GDN"),
    ("norms_misc", "norms", "Нормы, lm_head, роутер"),
    ("expert_h2d", "h2d", "H2D экспертов (не скрыто)"),
    ("cpu_gaps_gpu_idle", "idle", "Простой GPU"),
]
OTHER = ("other", "base", "Прочее")

NSYS = [
    ("nsys-decode16k", "decode_16k", "Шаг декода на контексте 16K по ступеням",
     "Nsight Systems, критический путь, мс на шаг. Меньше — лучше.", "ms_per_step", "мс/шаг"),
    ("nsys-prefill-short", "prefill_short", "Prefill короткого промпта по ступеням",
     "Nsight Systems, окно prefill на GPU, мс. Под профайлером абсолютные цифры завышены.", "window_ms", "мс"),
    ("nsys-prefill16k", "prefill_16k", "Prefill 16K по ступеням",
     "Nsight Systems, окно prefill на GPU, мс. Под профайлером абсолютные цифры завышены.", "window_ms", "мс"),
]


class LadderError(ValueError):
    pass


# ---------------------------------------------------------------- data

def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def median(xs):
    xs = sorted(x for x in (num(v) for v in xs or []) if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def len_label(n):
    n = int(n)
    if n < 1000:
        return str(n)
    return f"{n // 1024}K" if n % 1024 == 0 else f"{round(n / 1000)}K"


def validate(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise LadderError(f"schema must be {SCHEMA!r}")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise LadderError(f"schema_version must be {SCHEMA_VERSION}")
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        raise LadderError("steps must be a non-empty list")
    ids = []
    for s in steps:
        if not isinstance(s, dict) or not isinstance(s.get("id"), str) or not s["id"]:
            raise LadderError("every step needs a string id")
        ids.append(s["id"])
    if len(set(ids)) != len(ids):
        raise LadderError("step ids must be unique")
    for s in steps:
        p = s.get("parent")
        if p is not None and p not in ids:
            raise LadderError(f"step {s['id']}: unknown parent {p!r}")
        ref = get(s, "ab", "reference")
        if ref is not None and ref not in ids:
            raise LadderError(f"step {s['id']}: unknown ab.reference {ref!r}")
    lengths = doc.get("lengths")
    if lengths is not None and (not isinstance(lengths, list)
                                or not all(isinstance(n, int) and n > 0 for n in lengths)):
        raise LadderError("lengths must be a list of positive integers")
    if not any(s.get("cumulative", True) for s in steps):
        raise LadderError("at least one cumulative step is required")


class Ladder:
    def __init__(self, doc):
        validate(doc)
        self.doc = doc
        self.synthetic = bool(doc.get("synthetic"))
        self.steps = doc["steps"]
        self.by_id = {s["id"]: s for s in self.steps}
        self.cum = [s for s in self.steps if s.get("cumulative", True)]
        self.side = [s for s in self.steps if not s.get("cumulative", True)]
        self.order = self.cum + self.side
        self.lengths = [str(n) for n in (doc.get("lengths") or [64, 2048, 4096, 16384, 120000])]

    def arm(self, step, metric, length):
        return median(get(step, "ab", metric, length))

    def ratio(self, step, metric, length):
        """(ratio vs parent, how) — 'pairs' if paired against the parent, 'medians' otherwise."""
        parent = self.by_id.get(step.get("parent"))
        if parent is None:
            return None, None
        ab = step.get("ab") or {}
        if ab.get("reference") == parent["id"]:
            a, r = ab.get(metric, {}) or {}, ab.get("ref_" + metric, {}) or {}
            a, r = a.get(length) or [], r.get(length) or []
            pairs = [num(x) / num(y) for x, y in zip(a, r) if num(x) is not None and num(y)]
            if pairs and len(a) == len(r):
                return median(pairs), "pairs"
        x, y = self.arm(step, metric, length), self.arm(parent, metric, length)
        if x is not None and y:
            return x / y, "medians"
        return None, None

    def chain(self, metric, length):
        """[(step, before, after, ratio, how)] along the cumulative steps, from the L0 median."""
        base = self.arm(self.cum[0], metric, length)
        out, cur = [], base
        for s in self.cum[1:]:
            r, how = self.ratio(s, metric, length)
            after = cur * r if (cur is not None and r is not None) else cur
            out.append((s, cur, after, r, how))
            cur = after
        return base, out

    def release_ratio(self, metric, length):
        ra = self.doc.get("release_ab") or {}
        a, r = get(ra, metric, length) or [], get(ra, "ref_" + metric, length) or []
        pairs = [num(x) / num(y) for x, y in zip(a, r) if num(x) is not None and num(y)]
        return median(pairs) if pairs and len(a) == len(r) else None

    def short(self, step):
        return SHORT.get(step["id"]) or step.get("lever") or step.get("label") or step["id"]

    def label_x(self):
        """x of the lever caption next to the step id in row charts."""
        return 24 + max(len(s["id"]) for s in self.order) * 8.4 + 10


# ---------------------------------------------------------------- formatting

def nice_step(raw):
    """1, 2, 2.5 or 5 times a power of ten, at least raw."""
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def minus(s):
    return s.replace("-", "−")


def fmt_metric(metric, v):
    if v is None:
        return "н/д"
    if metric == "decode_tps":
        return f"{v:.1f}"
    return f"{v:.3f}" if v < 1 else f"{v:.2f}" if v < 10 else f"{v:.1f}"


def fmt_pct(r):
    return "н/д" if r is None else minus(f"{(r - 1) * 100:+.1f} %")


def good_colour(t, metric, r):
    if r is None or abs(r - 1) < 0.005:
        return t["muted"]
    better = r < 1 if metric == "ttft_s" else r > 1
    return t["moe"] if better else t["gather"]


def step_fill(t, lad, step):
    if not step.get("cumulative", True):
        return t["attn"]
    if step is lad.cum[0]:
        return t["base"]
    if step is lad.cum[-1]:
        return t["f1"]
    return t["fp8"]


def new_svg(lad, w, h, t, title, subtitle):
    s = rc.Svg(w, h, t, title, subtitle)
    if lad.synthetic:
        s.rect(w - 236, 18, 212, 26, t["gather"], rx=6, opacity=0.14, stroke=t["gather"])
        s.text(w - 130, 36, "СИНТЕТИКА · ЧЕРНОВИК", 12, t["gather"], anchor="middle", weight=700)
    return s


def save(s, outdir, name, theme):
    s.add("</svg>")
    path = os.path.join(outdir, f"{name}{'' if theme == 'light' else '-dark'}.svg")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(s.parts) + "\n")
    return path


def step_axis(s, lad, xs, y, t):
    for step, x in zip(lad.order, xs):
        s.text(x, y, step["id"], 12, t["fg"], anchor="middle", weight=600)
        s.text(x, y + 15, lad.short(step), 10.5, t["muted"], anchor="middle")


# ---------------------------------------------------------------- charts

def per_length_chart(lad, t, metric):
    """Small multiples: one row per prompt length, a bar per step, each row on its own scale."""
    ttft = metric == "ttft_s"
    W, rowh, top = 980, 112, 104
    H = top + rowh * len(lad.lengths) + 74
    title = "TTFT по ступеням лесенки" if ttft else "Скорость декода по ступеням лесенки"
    sub = ("Секунды до первого токена, медианы A/B. Меньше — лучше. У каждой строки свой масштаб."
           if ttft else "Токенов в секунду, медианы A/B. Больше — лучше. У каждой строки свой масштаб.")
    s = new_svg(lad, W, H, t, title, sub)
    x0, x1 = 120, W - 130
    n = len(lad.order)
    gap = 18 if lad.side else 0
    slot = (x1 - x0 - gap) / n
    bw = min(46, slot * 0.66)
    xs = [x0 + slot * (i + 0.5) + (gap if i >= len(lad.cum) else 0) for i in range(n)]
    s.legend(24, 88, [(t["base"], "L0 = f1"), (t["fp8"], "накопительные ступени"),
                      (t["f1"], "кандидат"), (t["attn"], "отдельные руки")], gap=18, size=12)
    s.text(W - 24, 88, "кандидат к L0", 12, t["muted"], anchor="end")
    for i, length in enumerate(lad.lengths):
        y = top + i * rowh
        if i % 2 == 1:
            s.rect(12, y - 2, W - 24, rowh, t["panel"], rx=6, opacity=0.55)
        s.text(24, y + 46, f"{len_label(length)}", 18, t["fg"], weight=700)
        s.text(24, y + 64, "токенов", 11, t["muted"])
        vals = [lad.arm(st, metric, length) for st in lad.order]
        vmax = max([v for v in vals if v is not None] or [1]) * 1.18
        base_y, hmax = y + rowh - 16, rowh - 44
        s.line(x0 - 8, base_y, x1 + 4, base_y, t["border"])
        for st, x, v in zip(lad.order, xs, vals):
            if v is None:
                s.text(x, base_y - 4, "н/д", 10.5, t["faint"], anchor="middle")
                continue
            h = hmax * v / vmax
            fill = step_fill(t, lad, st)
            s.rect(x - bw / 2, base_y - h, bw, h, fill, rx=3)
            s.text(x, base_y - h - 5, fmt_metric(metric, v), 11, t["fg"], anchor="middle")
        rel = lad.release_ratio(metric, length)
        how = f"релиз, {get(lad.doc, 'release_ab', 'pairs') or '?'} пар"
        if rel is None:
            b, ch = lad.chain(metric, length)
            ends = [c for c in ch if c[3] is not None]
            rel = (ch[-1][2] / b) if (b and ch and ch[-1][2] is not None and ends) else None
            how = f"цепочка, {len(ends)} из {len(ch)} ступеней"
        s.text(W - 24, y + 50, fmt_pct(rel), 17, good_colour(t, metric, rel), anchor="end", weight=700)
        s.text(W - 24, y + 66, how if rel is not None else "", 10.5, t["faint"], anchor="end")
    step_axis(s, lad, xs, top + rowh * len(lad.lengths) + 18, t)
    return s


def waterfall_chart(lad, t, metric, length):
    ttft = metric == "ttft_s"
    base, chain = lad.chain(metric, length)
    W, H = 980, 470
    if ttft:
        title = f"Вклад рычагов: TTFT короткого промпта, {len_label(length)} ток."
        unit = "с"
    else:
        title = f"Вклад рычагов: декод на контексте {len_label(length)}"
        unit = "ток/с"
    vals = [v for v in [base] + [c[2] for c in chain] + [c[1] for c in chain] if v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    cut = lo >= 0.5 * hi  # small deltas on a large base: cut the axis
    step = nice_step((hi - (lo * 0.85 if cut else 0)) / 5)
    vmin = step * int(lo * 0.85 / step) if cut else 0.0
    vmax = hi + (hi - vmin) * 0.22
    sub = ("Каждая ступень — медиана парных отношений к предыдущей ступени, накопленно от медианы L0. "
           + ("Меньше — лучше." if ttft else "Больше — лучше.") + (" Ось обрезана." if vmin > 0 else ""))
    s = new_svg(lad, W, H, t, title, sub)
    x0, x1, y0, y1 = 80, W - 40, 370, 110
    cols = 2 + len(chain)
    slot = (x1 - x0) / cols
    bw = min(56, slot * 0.62)
    if base is None:
        s.text(W / 2, (y0 + y1) / 2, "нет медианы L0 — водопад не строится", 15, t["faint"], anchor="middle")
        return s
    Y = lambda v: y0 - (y0 - y1) * (v - vmin) / (vmax - vmin)
    v = vmin
    while v <= vmax:
        s.line(x0, Y(v), x1, Y(v), t["grid"])
        s.text(x0 - 8, Y(v) + 4, f"{v:g}", 11, t["faint"], anchor="end")
        v = round(v + step, 6)
    s.text(x0 - 8, y1 - 14, unit, 11, t["faint"], anchor="end")
    if vmin > 0:
        s.add(f'<path d="M{x0 - 6},{y0 - 4} l6,-5 l6,5 l6,-5" fill="none" stroke="{t["muted"]}" stroke-width="1.5"/>')
    X = lambda i: x0 + slot * (i + 0.5)
    s.rect(X(0) - bw / 2, Y(base), bw, y0 - Y(base), t["base"], rx=3)
    s.text(X(0), Y(base) - 6, fmt_metric(metric, base), 12, t["fg"], anchor="middle", weight=600)
    labels = [(lad.cum[0]["id"], lad.short(lad.cum[0]))]
    medians_used = False
    prev_top = Y(base)
    for i, (st, before, after, r, how) in enumerate(chain, start=1):
        labels.append((st["id"], lad.short(st)))
        s.line(X(i - 1) + bw / 2, prev_top, X(i) - bw / 2, prev_top, t["faint"], dash="3 3")
        if r is None:
            s.text(X(i), Y(before) - 6, "н/д", 11, t["faint"], anchor="middle")
            prev_top = Y(before)
            continue
        medians_used |= how == "medians"
        colour = good_colour(t, metric, r)
        if colour == t["muted"]:
            colour = t["idle"]
        top, bot = Y(max(before, after)), Y(min(before, after))
        s.rect(X(i) - bw / 2, top, bw, max(bot - top, 1.5), colour, rx=2,
               dash="3 2" if how == "medians" else None, stroke=t["border"] if how == "medians" else None)
        d = after - before
        dtxt = minus(f"{d * 1000:+.0f} мс") if ttft else minus(f"{d:+.1f}")
        s.text(X(i), top - 20, dtxt, 11.5, t["fg"], anchor="middle", weight=600)
        s.text(X(i), top - 6, fmt_pct(r), 10.5, good_colour(t, metric, r), anchor="middle")
        prev_top = Y(after)
    end = chain[-1][2] if chain else base
    ie = len(chain) + 1
    s.line(X(ie - 1) + bw / 2, prev_top, X(ie) - bw / 2, prev_top, t["faint"], dash="3 3")
    s.rect(X(ie) - bw / 2, Y(end), bw, y0 - Y(end), t["f1"], rx=3)
    s.text(X(ie), Y(end) - 20, fmt_metric(metric, end), 12, t["fg"], anchor="middle", weight=600)
    s.text(X(ie), Y(end) - 6, fmt_pct(end / base), 10.5, good_colour(t, metric, end / base), anchor="middle")
    labels.append(("итог", "цепочка"))
    s.line(x0, y0, x1, y0, t["border"])
    for i, (a, b) in enumerate(labels):
        s.text(X(i), y0 + 18, a, 12, t["fg"], anchor="middle", weight=600)
        s.text(X(i), y0 + 33, b, 10.5, t["muted"], anchor="middle")
    rel = lad.release_ratio(metric, length)
    notes = []
    if rel is not None:
        notes.append(f"Релизный A/B кандидата против f1, 5 пар: {fmt_pct(rel)} (цепочка: {fmt_pct(end / base)}).")
    if medians_used:
        notes.append("Пунктир: пар против предыдущей ступени нет, взято отношение медиан.")
    if any(c[3] is None for c in chain):
        notes.append("н/д: ступень не измерена и в цепочку не вошла.")
    s.text(24, H - 16, " ".join(notes) or "Источник: results/ladder/summary.json.", 11, t["faint"])
    return s


def nsys_chart(lad, t, key, title, sub, total_key, unit):
    W, rowh, top = 980, 40, 104
    H = top + rowh * len(lad.order) + 96
    s = new_svg(lad, W, H, t, title, sub)
    x0, x1 = 150, W - 170
    totals = []
    for st in lad.order:
        sc = get(st, "nsys", key) or {}
        b = {k: num(v) for k, v in (sc.get("buckets_ms") or {}).items() if num(v) is not None}
        total = num(sc.get(total_key))
        if total is None and b:
            total = sum(b.values())
        totals.append((st, sc, b, total))
    vmax = max([tt for *_, tt in totals if tt] or [1]) * 1.04
    known = {k for k, _, _ in CATS}
    for i, (st, sc, b, total) in enumerate(totals):
        y = top + i * rowh
        if i == len(lad.cum) and lad.side:
            s.line(24, y - 4, W - 24, y - 4, t["border"], dash="4 3")
        s.text(24, y + 20, st["id"], 13, t["fg"], weight=600)
        s.text(lad.label_x(), y + 20, lad.short(st), 11.5, t["muted"])
        if total is None:
            s.text(x0, y + 20, "н/д", 12, t["faint"])
            continue
        x = x0
        span = (x1 - x0) / vmax
        if b:
            parts = [(k, ck) for k, ck, _ in CATS] + [(k, OTHER[1]) for k in sorted(b) if k not in known]
            for k, ck in parts:
                w = b.get(k, 0) * span
                if w <= 0:
                    continue
                s.rect(x, y + 6, w, 24, t[ck])
                if w >= 34:
                    s.text(x + w / 2, y + 22, f"{b[k]:.1f}" if b[k] < 10 else f"{b[k]:.0f}", 10.5,
                           rc.on(t, t[ck]), anchor="middle", weight=600)
                x += w
        else:
            w = total * span
            s.rect(x, y + 6, w, 24, t["na"], stroke=t["border"], dash="3 2")
            s.text(x + 8, y + 22, "без разбивки", 10.5, t["muted"])
            x += w
        tt = f"{total:.2f}" if total < 100 else f"{total:.0f}"
        s.text(x + 8, y + 23, f"{tt} {unit}", 12, t["fg"], weight=600)
        ov = num(sc.get("h2d_overlap_pct"))
        if ov is not None:
            s.text(W - 24, y + 23, f"H2D скрыто {ov:.0f} %", 11, t["muted"], anchor="end")
    ly = top + rowh * len(lad.order) + 22
    s.legend(24, ly, [(t[c], lab) for _, c, lab in CATS[:4]], gap=22, size=12)
    s.legend(24, ly + 22, [(t[c], lab) for _, c, lab in CATS[4:]], gap=22, size=12)
    s.text(24, H - 14, "Критический путь: перекрытые копии в сумму не входят. Ключи корзин — как в "
           "results/nsight/dashboard.json.", 11, t["faint"])
    return s


def memory_chart(lad, t):
    total_mib = num(lad.doc.get("gpu_total_mib"))
    limit = total_mib - WINDOWS_RESERVE_MIB if total_mib else None
    cols = [
        ("expert_slots", "Слоты экспертов", lambda m: num(m.get("expert_slots")), "{:.0f}", None),
        ("process_peak_mib", "Пик процесса, МиБ", lambda m: num(m.get("process_peak_mib")), "{:.0f}", limit),
        ("free_vram_min_mib", "Мин. свободно, МиБ", lambda m: num(m.get("free_vram_min_mib")), "{:.0f}", None),
        ("misses", "Промахов на токен", lambda m: num(get(m, "misses_per_token", "16384"))
         if num(get(m, "misses_per_token", "16384")) is not None else num(get(m, "misses_per_token", "all")),
         "{:.1f}", None),
    ]
    W, rowh, top = 980, 36, 128
    H = top + rowh * len(lad.order) + 70
    sub = "Видеопамять, слоты кэша экспертов и промахи декода (16K, иначе медиана по всем длинам)."
    s = new_svg(lad, W, H, t, "Память по ступеням", sub)
    x0 = 190
    cw = (W - 24 - x0) / len(cols)
    for j, (_, name, _, _, _) in enumerate(cols):
        s.text(x0 + j * cw, top - 22, name, 12, t["muted"], weight=600)
    s.text(24, top - 22, "Ступень / ratio", 12, t["muted"], weight=600)
    s.line(24, top - 12, W - 24, top - 12, t["border"])
    for j, (_, _, fn, fmt, lim) in enumerate(cols):
        vals = [fn(st.get("memory") or {}) for st in lad.order]
        vmax = max([v for v in vals if v is not None] + ([lim] if lim else []) or [1])
        bwmax = cw - 70
        cx = x0 + j * cw
        for i, (st, v) in enumerate(zip(lad.order, vals)):
            y = top + i * rowh
            if v is None:
                s.text(cx, y + 18, "н/д", 11, t["faint"])
                continue
            w = bwmax * v / vmax
            s.rect(cx, y + 6, w, 18, step_fill(t, lad, st), rx=3)
            s.text(cx + w + 6, y + 19, fmt.format(v), 11.5, t["fg"])
        if lim:
            lx = cx + bwmax * lim / vmax
            s.line(lx, top - 8, lx, top + rowh * len(lad.order), t["gather"], sw=1.5, dash="4 3")
            s.text(lx, top + rowh * len(lad.order) + 14, f"предел {lim:.0f}", 10.5, t["gather"], anchor="middle")
    for i, st in enumerate(lad.order):
        y = top + i * rowh
        s.text(24, y + 19, st["id"], 13, t["fg"], weight=600)
        mr = num(get(st, "memory", "memory_ratio"))
        s.text(lad.label_x(), y + 19, lad.short(st) + (f" · {mr:g}" if mr is not None else ""), 11, t["muted"])
    note = ("Правило владельца: 1 ГиБ VRAM — Windows, остальное — модели. Предел = всего VRAM − 1 ГиБ; "
            "гейт падает на подкачке и замедлении, не на пороге свободной памяти.")
    s.text(24, H - 16, note, 11, t["faint"])
    return s


def teacher_chart(lad, t):
    n = len(lad.order)
    W, H = 980, 214
    s = new_svg(lad, W, H, t, "Teacher-гейт по ступеням",
                "125 промптов teacher forcing, бит-точность: нужно 125/125 и max_abs 0.")
    gap = 8
    bw = (W - 48 - gap * (n - 1)) / n
    for i, st in enumerate(lad.order):
        x, y = 24 + i * (bw + gap), 84
        te = st.get("teacher") or {}
        status = te.get("status") or "not_run"
        passed, total, mx = num(te.get("passed")), num(te.get("total")), num(te.get("max_abs"))
        if status == "pass" and not (passed is not None and passed == total and mx == 0):
            status = "fail"  # a pass needs 125/125 and max_abs 0
        fill = {"pass": t["moe"], "fail": t["gather"]}.get(status, t["na"])
        s.rect(x, y, bw, 96, fill, rx=8, stroke=t["border"],
               dash="4 3" if status in ("pending", "not_run") else None)
        fg = rc.on(t, fill) if status in ("pass", "fail") else t["muted"]
        s.text(x + bw / 2, y + 20, st["id"], 13, fg, anchor="middle", weight=700)
        s.text(x + bw / 2, y + 35, lad.short(st), 10, fg, anchor="middle")
        word = {"pass": "PASS", "fail": "FAIL", "pending": "идёт"}.get(status, "не было")
        s.text(x + bw / 2, y + 60, word, 16 if status in ("pass", "fail") else 13, fg, anchor="middle", weight=700)
        if passed is not None and total is not None:
            s.text(x + bw / 2, y + 76, f"{passed:.0f}/{total:.0f}", 11, fg, anchor="middle")
        if mx is not None:
            s.text(x + bw / 2, y + 89, f"max_abs {mx:g}", 10, fg, anchor="middle")
    s.text(24, H - 14, "PASS засчитывается только при 125/125 и max_abs 0; иначе плашка красная.", 11, t["faint"])
    return s


# ---------------------------------------------------------------- tables

def tables(lad, lang):
    ru = lang == "ru"
    out = []
    if lad.synthetic:
        out.append("> **" + ("СИНТЕТИКА, ЧЕРНОВИК: не публиковать как результат."
                             if ru else "SYNTHETIC DRAFT: do not publish as a result.") + "**\n")
    heads = [len_label(n) for n in lad.lengths]
    for metric, name in (("ttft_s", "TTFT, с" if ru else "TTFT, s"),
                         ("decode_tps", "Декод, ток/с" if ru else "Decode, tok/s")):
        out.append(f"### {name}\n")
        out.append("| " + ("Ступень" if ru else "Step") + " | " + ("Рычаг" if ru else "Lever") + " | "
                   + " | ".join(heads) + " |")
        out.append("|---|---|" + "---|" * len(heads))
        for st in lad.order:
            cells = []
            for length in lad.lengths:
                v = lad.arm(st, metric, length)
                r, how = lad.ratio(st, metric, length)
                c = fmt_metric(metric, v) if v is not None else ("н/д" if ru else "n/a")
                if r is not None:
                    c += f" ({fmt_pct(r)}{'' if how == 'pairs' else '*'})"
                cells.append(c.replace("н/д", "н/д" if ru else "n/a"))
            out.append(f"| {st['id']} | {st.get('label') or lad.short(st)} | " + " | ".join(cells) + " |")
        out.append("")
    out.append("### " + ("Teacher и память" if ru else "Teacher and memory") + "\n")
    out.append("| " + (" | ".join(["Ступень", "Teacher", "memory_ratio", "Слоты", "Пик процесса, МиБ",
                                     "Мин. свободно, МиБ", "Промахов на токен"]) if ru else
                       " | ".join(["Step", "Teacher", "memory_ratio", "Slots", "Process peak, MiB",
                                   "Min free, MiB", "Misses per token"])) + " |")
    out.append("|---|---|---|---|---|---|---|")
    na = "н/д" if ru else "n/a"
    for st in lad.order:
        te = st.get("teacher") or {}
        m = st.get("memory") or {}
        tt = te.get("status") or "not_run"
        if num(te.get("passed")) is not None:
            tt += f" {te['passed']}/{te.get('total')}, max_abs {te.get('max_abs')}"
        miss = get(m, "misses_per_token", "16384")
        miss = miss if num(miss) is not None else get(m, "misses_per_token", "all")
        row = [st["id"], tt] + [na if num(x) is None else f"{x:g}" for x in
                                (m.get("memory_ratio"), m.get("expert_slots"), m.get("process_peak_mib"),
                                 m.get("free_vram_min_mib"))] + [na if num(miss) is None else f"{miss:.1f}"]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append(("В скобках — вклад ступени: медиана парных отношений к родительской ступени; "
                "`*` — отношение медиан (пар нет).") if ru else
               ("In brackets: the step's contribution, the median of paired ratios against its parent; "
                "`*` marks a ratio of medians (no pairs)."))
    ra = lad.doc.get("release_ab") or {}
    if ra:
        out.append("")
        out.append(("Релизный A/B {} против {}, {} пар, вердикт {}:" if ru else
                    "Release A/B {} vs {}, {} pairs, verdict {}:").format(
            ra.get("candidate"), ra.get("reference"), ra.get("pairs"), ra.get("verdict")))
        for metric in ("ttft_s", "decode_tps"):
            vals = ", ".join(f"{len_label(n)} {fmt_pct(lad.release_ratio(metric, n))}" for n in lad.lengths)
            out.append(f"- {metric}: {vals}")
    return "\n".join(out).replace("н/д", na) + "\n"


# ---------------------------------------------------------------- main

def render(lad, outdir):
    os.makedirs(outdir, exist_ok=True)
    written = []
    short_len = lad.lengths[0]
    decode_len = "16384" if "16384" in lad.lengths else lad.lengths[-1]
    for theme, t in THEMES.items():
        written.append(save(per_length_chart(lad, t, "ttft_s"), outdir, "ttft-steps", theme))
        written.append(save(per_length_chart(lad, t, "decode_tps"), outdir, "decode-steps", theme))
        written.append(save(waterfall_chart(lad, t, "ttft_s", short_len), outdir, "waterfall-ttft-short", theme))
        written.append(save(waterfall_chart(lad, t, "decode_tps", decode_len), outdir, "waterfall-decode16k", theme))
        for name, key, title, sub, total_key, unit in NSYS:
            written.append(save(nsys_chart(lad, t, key, title, sub, total_key, unit), outdir, name, theme))
        written.append(save(memory_chart(lad, t), outdir, "memory-steps", theme))
        written.append(save(teacher_chart(lad, t), outdir, "teacher-steps", theme))
    for lang, name in (("ru", "ladder-table.md"),):
        path = os.path.join(outdir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(tables(lad, lang))
        written.append(path)
    if lad.synthetic:
        path = os.path.join(outdir, "DRAFT.md")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Черновик графиков лесенки main128 — СИНТЕТИКА\n\n"
                    "Отрисовано по синтетической фикстуре, не по прогонам лесенки. Не результат, не публиковать.\n"
                    "Финальные графики пишутся в `results/ladder/charts/` из `results/ladder/summary.json`:\n\n"
                    "    python3 tools/ladder-charts.py\n\n"
                    + (f"Фикстура: {lad.doc['fixture_note']}\n" if lad.doc.get("fixture_note") else ""))
        written.append(path)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=os.path.join(LADDER, "summary.json"))
    p.add_argument("--output", help="default: results/ladder/charts, or charts-draft for synthetic input")
    a = p.parse_args(argv)
    try:
        with open(a.input, encoding="utf-8") as f:
            lad = Ladder(json.load(f))
    except (OSError, json.JSONDecodeError, LadderError) as e:
        print(f"ladder-charts: {e}", file=sys.stderr)
        return 2
    final = os.path.join(LADDER, "charts")
    out = a.output or (os.path.join(LADDER, "charts-draft") if lad.synthetic else final)
    if lad.synthetic and os.path.normcase(os.path.realpath(out)) == os.path.normcase(os.path.realpath(final)):
        print("ladder-charts: synthetic input is never written to results/ladder/charts; "
              "use results/ladder/charts-draft", file=sys.stderr)
        return 2
    for path in render(lad, out):
        print(os.path.relpath(path, ROOT) if path.startswith(ROOT) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
