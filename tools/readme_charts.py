#!/usr/bin/env python3
"""Generate the README charts as light and dark SVG files in docs/img/.

No third-party packages. Three series are shown: the original FreeToken (bf128), the f1 release
(14.10.2026) and the main128 release (16.10.2026).

- The old matrix of 13-14.10 (bf128 and f1) is copied from the README of the f1 release (OLD below).
- "f1" is the previous release (the installed f1 runtime). "f1L0" is the f1 settings on the new
  code (profile main-release in the A/B of 16.10); they are different things and are never mixed.
- main128 is NOT a direct measurement against the original. It is chained: the f1 value of the old
  matrix times the main128/f1 ratio of two consecutive matrices of one session on 16.10 (f1:
  results/ladder/ab-f1old.json, main128: results/ladder/ab-f1new.json, 3 pairs each, medians).
  The 64K row is a direct three-point measurement of its own (results/ladder/64k.json: f1, f1L0 and
  main128, 3 pairs, fresh 65536); the f1 point there pools the profile and its alias (n = 6).
- Nsight Systems numbers of main128 are read from results/ladder/nsys.json (step L9); the Nsight
  Compute charts still show f1, because Nsight Compute could not profile the model
  (results/ladder/ncu.json).

    python tools/readme_charts.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
NSIGHT = os.path.join(ROOT, "results", "nsight")
LADDER = os.path.join(ROOT, "results", "ladder")
MATRIX_DIR = os.path.join(ROOT, "results", "matrix")

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"

THEMES = {
    "light": {
        "bg": "#ffffff", "card": "#f6f8fa", "border": "#d0d7de", "fg": "#1f2328",
        "muted": "#59636e", "faint": "#8c959f", "grid": "#e1e6eb", "panel": "#ffffff",
        "base": "#a8b1ba", "f1": "#4d8c00", "f1text": "#3d7000",
        "m128": "#0b6bcb", "m128text": "#0a58a8",
        "attn": "#8250df", "moe": "#1a7f37", "fp8": "#0969da", "gdn": "#1b7c83",
        "norms": "#bf3989", "h2d": "#d1561a", "gather": "#cf222e", "idle": "#d0d7de",
        "host": "#eaeef2", "heat": ["#f2f5f8", "#fde9b6", "#f9b46a", "#ec6a2c", "#c5221f"],
        "na": "#eaeef2",
    },
    "dark": {
        "bg": "#0d1117", "card": "#161b22", "border": "#30363d", "fg": "#e6edf3",
        "muted": "#9198a1", "faint": "#6e7681", "grid": "#262c34", "panel": "#0d1117",
        "base": "#4f5864", "f1": "#76b900", "f1text": "#8fd400",
        "m128": "#58a6ff", "m128text": "#79b8ff",
        "attn": "#a371f7", "moe": "#3fb950", "fp8": "#4493f8", "gdn": "#39c5cf",
        "norms": "#db61a2", "h2d": "#f0883e", "gather": "#ff6a69", "idle": "#3d444d",
        "host": "#21262d", "heat": ["#1c2128", "#4a3b14", "#8a4b12", "#d2560f", "#f2453d"],
        "na": "#21262d",
    },
}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lum(h):
    r, g, b = (c / 255 for c in hexrgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def heat(t, v):
    """Colour for a load fraction v in [0, 1] on the theme's heat scale."""
    stops = t["heat"]
    v = max(0.0, min(1.0, v))
    pos = v * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    a, b = hexrgb(stops[i]), hexrgb(stops[i + 1])
    return "#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


def on(t, fill):
    """Readable text colour on a filled shape."""
    return "#ffffff" if lum(fill) < 0.5 else "#1f2328"


class Svg:
    def __init__(self, w, h, t, title, subtitle=None):
        self.w, self.h, self.t = w, h, t
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" role="img" aria-label="{esc(title)}">',
            f'<title>{esc(title)}</title>',
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12" fill="{t["card"]}" stroke="{t["border"]}"/>',
        ]
        self.text(24, 38, title, 19, t["fg"], weight=600)
        if subtitle:
            self.text(24, 61, subtitle, 13, t["muted"])

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, fill=None, anchor="start", weight=None, italic=False):
        w = f' font-weight="{weight}"' if weight else ""
        i = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill or self.t["fg"]}" '
                 f'text-anchor="{anchor}"{w}{i}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, sw=1, opacity=None, dash=None):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}"{o}{s}{d}/>')

    def line(self, x1, y1, x2, y2, stroke, sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d}/>')

    def legend(self, x, y, items, gap=22, size=13):
        """items: [(colour, label)], laid out left to right; returns end x."""
        for colour, label in items:
            self.rect(x, y - 10, 12, 12, colour, rx=3)
            self.text(x + 18, y, label, size, self.t["muted"])
            x += 18 + len(label) * size * 0.56 + gap
        return x

    def save(self, name, theme):
        self.add("</svg>")
        suffix = "" if theme == "light" else "-dark"
        path = os.path.join(OUT, f"{name}{suffix}.svg")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.parts) + "\n")
        return path


# ---------------------------------------------------------------- data

def jload(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


# The old matrix of the f1 release (13-14.10.2026): medians of 5 paired rounds, 128 output tokens,
# one session (results/matrix/final-f1-m-main-release-vs-bf128-comparison.json).
# (key of the A/B case, label lines, case, length, TTFT bf128, TTFT f1, decode bf128, decode f1)
OLD = [
    ("fresh_64", ["Промпт 64 токена"], "fresh", 64, 0.75, 0.79, 96.4, 122.8),
    ("fresh_4096", ["Свежий промпт 4K"], "fresh", 4096, 2.24, 0.77, 98.9, 124.5),
    ("fresh_16384", ["Свежий промпт 16K"], "fresh", 16384, 6.78, 1.59, 94.2, 119.1),
    ("fresh_65536", ["Свежий промпт 64K"], "fresh", 65536, 28.46, 9.14, 84.4, 99.3),
    ("fresh_120000", ["Свежий промпт 120K"], "fresh", 120000, 54.77, 22.07, 80.7, 94.6),
    ("history_continuation_4096", ["Продолжение диалога, 4K"], "history_continuation", 4096, 2.20, 0.78, 106.4, 126.9),
    ("last_message_edit_16384", ["Правка последнего", "сообщения, 16K"], "last_message_edit", 16384, 6.85, 0.78, 90.4, 112.4),
    ("tool_result_edit_16384", ["Правка результата", "инструмента, 16K"], "tool_result_edit", 16384, 7.11, 0.76, 94.9, 113.4),
]

# Section 2 of the README: one session at context 65536, 192 output tokens, 2 rounds (13-14.10).
# (label, key of the main128/f1 ratio, original TTFT/decode, +FP32 TTFT/decode, f1 TTFT/decode)
SEC2 = [
    ("4K", "fresh_4096", (1.93, 116.0), (1.85, 118.1), (0.58, 142.4)),
    ("16K", "fresh_16384", (6.04, 111.3), (5.86, 115.7), (1.49, 138.0)),
    ("60K‡", "fresh_65536", (22.86, 101.3), (22.43, 107.6), (7.50, 122.2)),
]

# README section 3 (long256): original mechanics (13.10, Triton, chunk 4096) and f1 (14.10).
# (original, f1)
LONG_OLD = {
    "ttft250": (126.5, 77.4),       # original single run; f1 median of 11
    "dec250": (None, 60.8),
    "ttft261": (139.5, 91.3),       # original median of 5; f1 single run
    "dec261": (52.5, 52.5),
}


def _ab_cases():
    """f1 (installed runtime, matrix s5-f1old), main128 and f1L0 (matrix s5-f1new): consecutive, one session."""
    old = jload(LADDER, "ab-f1old.json")
    new = jload(LADDER, "ab-f1new.json")
    return old["steps"][0]["cases"], new["steps"][1]["cases"], new["steps"][0]["cases"]


def release_matrix():
    """Rows of the three-way matrix. All main128 values are chained (see the module docstring)."""
    f1c, mc, _ = _ab_cases()
    cmp_old = jload(MATRIX_DIR, "final-f1-m-main-release-vs-bf128-comparison.json")
    k64 = jload(LADDER, "64k.json")["points"]
    p64 = {name.split(" ")[0].split("_")[0]: pt for name, pt in k64.items()}   # f1, f1L0, main128

    def gm(cmp, case, length):
        for r in cmp["rows"]:
            if r["case"] == case and r["length"] == length and r["metric"] == "elapsed_s":
                return r["geomean"]
        return None

    out = []
    for key, label, case, length, bt, ft, bd, fd in OLD:
        if key in f1c:
            t_f1, t_m = f1c[key]["ttft_s"]["median"], mc[key]["ttft_s"]["median"]
            d_f1, d_m = f1c[key]["decode_tps"]["median"], mc[key]["decode_tps"]["median"]
            src = "A/B"
            # whole request: ratio of medians of two consecutive matrices (not paired, no interval)
            e_new = f1c[key]["elapsed_s"]["median"] / mc[key]["elapsed_s"]["median"]
        else:                                   # 64K: its own measurement, three points, 3 pairs
            a, b = p64["f1"], p64["main128"]
            t_f1, t_m = a["ttft_s"]["median"], b["ttft_s"]["median"]
            d_f1, d_m = a["decode_tps"]["median"], b["decode_tps"]["median"]
            src = "64K"
            e_new = a["elapsed_s"]["median"] / b["elapsed_s"]["median"]
        e_old = gm(cmp_old, case, length)
        mt = ft * t_m / t_f1                    # main128 TTFT, chained
        md = fd * d_m / d_f1
        out.append(dict(
            key=key, label=label, src=src, case=case, length=length,
            ttft_orig=bt, ttft_f1=ft, ttft_m=mt,
            dec_orig=bd, dec_f1=fd, dec_m=md,
            ttft_x_f1=t_f1 / t_m, ttft_x_orig=bt / mt,
            dec_x_f1=d_m / d_f1, dec_x_orig=md / bd,
            el_f1_vs_orig=e_old, el_m_vs_f1=e_new,
            el_m_vs_orig=(e_old * e_new) if (e_old and e_new) else None,
            today=dict(ttft_f1=t_f1, ttft_m=t_m, dec_f1=d_f1, dec_m=d_m)))
    return out, dict(old=cmp_old["tool_task"], new=jload(LADDER, "release-ab-compare.json")["tool_task"])


def sec2_rows():
    by = {r["key"]: r for r in release_matrix()[0]}
    rows = []
    for label, key, orig, fp32, f1 in SEC2:
        r = by[key]
        t_ratio = r["today"]["ttft_m"] / r["today"]["ttft_f1"]
        d_ratio = r["today"]["dec_m"] / r["today"]["dec_f1"]
        m = (f1[0] * t_ratio, f1[1] * d_ratio)
        rows.append(dict(label=label, orig=orig, fp32=fp32, f1=f1, m=m,
                         ttft_x_f1=1 / t_ratio, dec_x_f1=d_ratio,
                         ttft_x_orig=orig[0] / m[0], dec_x_orig=m[1] / orig[1]))
    return rows


def long_rows():
    c = jload(LADDER, "long256-candidate.json")
    l = jload(LADDER, "long256.json")
    return dict(
        ttft250_m=c["fresh"]["250000"]["ttft_s"]["median"], dec250_m=c["fresh"]["250000"]["decode_tps"]["median"],
        ttft250_l=l["screen"]["fresh"]["250000"]["ttft_s_median"], dec250_l=l["screen"]["fresh"]["250000"]["decode_tps_median"],
        ttft261_m=c["boundary"]["ttft_s"], dec261_m=c["boundary"]["decode_tps"],
        ttft261_l=l["boundary"]["ttft_s"], dec261_l=l["boundary"]["decode_tps"],
        ttft120_m=c["fresh"]["120000"]["ttft_s"]["median"], dec120_m=c["fresh"]["120000"]["decode_tps"]["median"],
        ttft120_l=l["screen"]["fresh"]["120000"]["ttft_s_median"], dec120_l=l["screen"]["fresh"]["120000"]["decode_tps_median"])


def length_series():
    """x, original, f1, main128 for the TTFT-by-length chart.

    4K-120K: the old matrix and the chained main128 (64K from its own three-point measurement).
    250K: the original mechanics (13.10), f1 (median of 11) and a DIRECT measurement of the
    full main128 levers on the long256 geometry (results/ladder/long256-candidate.json)."""
    rows = {r["key"]: r for r in release_matrix()[0]}
    keys = ["fresh_4096", "fresh_16384", "fresh_65536", "fresh_120000"]
    x = [4096, 16384, 65536, 120000, 250000]
    base = [rows[k]["ttft_orig"] for k in keys] + [LONG_OLD["ttft250"][0]]
    f1 = [rows[k]["ttft_f1"] for k in keys] + [LONG_OLD["ttft250"][1]]
    m = [rows[k]["ttft_m"] for k in keys] + [long_rows()["ttft250_m"]]
    return x, base, f1, m


def decode_length_series():
    """x, original, f1, main128 decode (tok/s) for the decode-by-length chart; None = no real data.

    64-120K: the old matrix and the chained main128 (64K from its own three-point measurement).
    250K: f1 (median of 11) and a DIRECT measurement of the full main128 levers on the long256
    geometry (results/ladder/long256-candidate.json). The original mechanics have no decode
    measurement at 250K, the point is not drawn."""
    rows = {r["key"]: r for r in release_matrix()[0]}
    keys = ["fresh_64", "fresh_4096", "fresh_16384", "fresh_65536", "fresh_120000"]
    x = [64, 4096, 16384, 65536, 120000, 250000]
    base = [rows[k]["dec_orig"] for k in keys] + [LONG_OLD["dec250"][0]]
    f1 = [rows[k]["dec_f1"] for k in keys] + [LONG_OLD["dec250"][1]]
    m = [rows[k]["dec_m"] for k in keys] + [long_rows()["dec250_m"]]
    return x, base, f1, m


CATS = [
    ("full_attention", "attn", "Attention"),
    ("moe_experts", "moe", "Эксперты MoE"),
    ("fp8_gemv_gemm", "fp8", "Плотные слои FP8"),
    ("gdn", "gdn", "GDN"),
    ("norms_misc", "norms", "Нормы, lm_head, прочее"),
    ("expert_h2d", "h2d", "Копирование экспертов (не скрыто)"),
    ("decode_gather", "gather", "Подкачка экспертов в decode"),
    ("cpu_gaps_gpu_idle", "idle", "Простой GPU"),
]

# results/ladder/nsys.json bucket -> chart category (copies_other joins the "norms and misc" bucket)
NSYS_MAP = {
    "attention": "full_attention", "moe": "moe_experts", "fp8_dense_oproj": "fp8_gemv_gemm", "gdn": "gdn",
    "norm_sampling_misc": "norms_misc", "copies_other": "norms_misc", "h2d_exposed": "expert_h2d",
    "gather": "decode_gather", "idle": "cpu_gaps_gpu_idle",
}

# (request in nsys.json, phase, title)
SCENARIOS = [
    ("warm_decode", "prefill", "Короткий промпт, 278 токенов"),
    ("fresh_16k", "prefill", "Prefill 16K"),
    ("fresh_120k", "prefill", "Prefill 120K"),
    ("fresh_16k", "decode", "Decode на контексте 16K"),
    ("fresh_120k", "decode", "Decode на контексте 120K"),
]


def nsys_scenario(step_id, request, phase):
    """Bucket milliseconds mapped to CATS keys, the critical total (ms) and the transport block."""
    nsys = jload(LADDER, "nsys.json")
    step = next(s for s in nsys["steps"] if s["id"] == step_id)
    r = step["requests"][request]
    if phase == "prefill":
        raw, total = r["prefill_buckets_ms"], r["prefill_critical_ms"]
    else:
        raw, total = r["decode_buckets_ms_per_step"], r["decode_step_critical_ms"]
    b = {}
    for k, v in raw.items():
        b[NSYS_MAP[k]] = b.get(NSYS_MAP[k], 0.0) + v
    return b, total, r["transport"][phase]


# README «Память и KV-кэш»: soak of main128 (telemetry of the 2 h soak), GiB.
# Slots: 4603 x 1.782 MB. Dense: without the embedding table (0.95 GiB moved to host RAM).
# "CUDA, cuBLAS, desktop" is the remainder of the 15.92 GiB card; free is the soak median.
VRAM = [
    ("moe", "Кэш экспертов, 4603 слота", 7.64),
    ("fp8", "Плотные веса и масштабы", 1.57),
    ("attn", "KV-кэш BF16, 131K токенов", 2.50),
    ("gdn", "Состояния GDN", 0.54),
    ("norms", "Прочее PyTorch", 0.56),
    ("base", "CUDA, cuBLAS, рабочий стол", 2.02),
    ("idle", "Свободно (медиана)", 1.09),
]

# Ladder (results/ladder/summary.json): step id in the data, label of the lever for the waterfall charts.
LADDER_STEPS = [
    ("F1", "f1 (прошлый выпуск)"), ("L0", "f1L0: f1 на новом коде"),
    ("L1", "короткий маршрут prefill"), ("L2", "CUDA-ядра GDN для prefill"),
    ("L3", "shared-эксперт параллельно с подкачкой"), ("L4", "эмбеддинги в RAM хоста"),
    ("L5", "больше слотов экспертов"), ("L6", "бит-точные ядра декода"),
    ("L7", "копирование без ожидания GEMM"), ("L8", "короткие куски: только выбранные эксперты"),
    ("L9", "слитый эпилог MoE-блока"),
]


def ladder_medians(kind, length):
    """Medians of the ladder steps L0..L9 (3 pairs each): kind 'ttft_s' or 'decode_tps'."""
    summ = jload(LADDER, "summary.json")
    by = {s["id"]: s for s in summ["steps"]}
    f1 = jload(LADDER, "ab-f1old.json")["steps"][0]["cases"]["fresh_%d" % length][kind]["median"]
    return [f1 if i == "F1" else by[i]["ab"]["stats"][kind][str(length)]["median"] for i, _ in LADDER_STEPS]


def load_nsight():
    with open(os.path.join(NSIGHT, "dashboard.json"), encoding="utf-8") as f:
        dash = json.load(f)
    with open(os.path.join(NSIGHT, "ncu-component-kernels.json"), encoding="utf-8") as f:
        ncu = json.load(f)["kernels"]
    return dash, ncu


def ncu_find(ncu, prefix):
    for k in ncu:
        if k["label"].startswith(prefix):
            return k
    raise KeyError(prefix)


# ---------------------------------------------------------------- charts

def fmt_x(v):
    if v >= 10:
        return f"×{v:.0f}"
    if v >= 2:
        return f"×{v:.1f}"
    return f"×{v:.2f}"


def paired_bars(t, kind):
    """Horizontal bars, three series per case: original FreeToken, f1, main128 (TTFT or decode)."""
    rows = release_matrix()[0]
    W, rowh, top = 1000, 66, 124
    H = top + rowh * len(rows) + 78
    if kind == "ttft":
        title = "Время до первого токена: исходный FreeToken, f1 и main128"
        sub = "Матрица 128K, медианы, секунды. Меньше — лучше. main128 — приведённые значения (см. примечание)."
        vmax, ticks, unit = 60, [0, 10, 20, 30, 40, 50, 60], "с"
    else:
        title = "Скорость генерации: исходный FreeToken, f1 и main128"
        sub = "Та же матрица, ответ 128 токенов, токенов в секунду. Больше — лучше. main128 — приведённые значения."
        vmax, ticks, unit = 160, [0, 20, 40, 60, 80, 100, 120, 140, 160], "ток/с"
    s = Svg(W, H, t, title, sub)
    x0, x1 = 230, W - 290
    span = x1 - x0
    s.legend(24, 92, [(t["base"], "Исходный FreeToken (bf128)"), (t["f1"], "f1, прошлый выпуск"), (t["m128"], "main128, 16.10")])
    xa, xb = W - 150, W - 24
    s.text(xa, 92, "main128 к исходному", 12, t["muted"], anchor="end")
    s.text(xb, 92, "к f1", 12, t["muted"], anchor="end")
    bottom = top + rowh * len(rows)
    for v in ticks:
        x = x0 + span * v / vmax
        s.line(x, top - 6, x, bottom, t["grid"])
        s.text(x, bottom + 18, f"{v}", 12, t["faint"], anchor="middle")
    s.text(x1 + 12, bottom + 18, unit, 12, t["faint"])
    for i, r in enumerate(rows):
        y = top + i * rowh
        if i % 2 == 1:
            s.rect(12, y - 4, W - 24, rowh, t["panel"], rx=6, opacity=0.55)
        label = r["label"]
        ly = y + rowh / 2 - (len(label) - 1) * 8 + 4
        for j, part in enumerate(label):
            s.text(24, ly + j * 16, part, 14, t["fg"])
        if kind == "ttft":
            vals = (r["ttft_orig"], r["ttft_f1"], r["ttft_m"])
            b1, b2 = fmt_x(r["ttft_x_orig"]), fmt_x(r["ttft_x_f1"])
        else:
            vals = (r["dec_orig"], r["dec_f1"], r["dec_m"])
            b1, b2 = f"+{(r['dec_x_orig'] - 1) * 100:.0f}%", f"+{(r['dec_x_f1'] - 1) * 100:.0f}%"
        cols = (t["base"], t["f1"], t["m128"])
        for k, (v, colour) in enumerate(zip(vals, cols)):
            by = y + 7 + k * 17
            w = span * v / vmax
            s.rect(x0, by, w, 13, colour, rx=3)
            txt = f"{v:.2f}" if kind == "ttft" else f"{v:.1f}"
            fill = (t["muted"], t["fg"], t["m128text"])[k]
            s.text(x0 + w + 6, by + 11, txt, 12, fill, weight=600 if k else None)
        s.text(xa, y + rowh / 2 + 6, b1, 17, t["fg"], anchor="end", weight=700)
        s.text(xb, y + rowh / 2 + 6, b2, 17, t["m128text"], anchor="end", weight=700)
    s.text(24, H - 40, "main128 здесь не прямой замер против исходного: значение f1 из матрицы 13–14.10 × отношение main128/f1 из двух матриц подряд "
           "16.10 (по 3 пары).", 11, t["faint"])
    s.text(24, H - 24, "Множитель к исходному = (исходный / f1 в матрице 13–14.10) × (f1 / main128 16.10). 64K — отдельный прямой замер 16.10 "
           "(f1, f1L0, main128, 3 пары).", 11, t["faint"])
    s.text(24, H - 8, "Короткие случаи: в матрице 13.10 не было прогрева 60 с, f1 там медленнее (64 токена: 0.79 против 0.41 с), поэтому приведённые "
           "значения — оценка; прямой замер main128 после прогрева — 0.200 с.", 11, t["faint"])
    return s


def length_chart(t):
    x_, base, f1, m = length_series()
    W, H = 980, 612
    s = Svg(W, H, t, "TTFT по длине свежего промпта",
            "Секунды до первого токена. 4K–120K — матрица 128K (main128 приведён), 250K — профиль long256.")
    x0, x1, y0, y1 = 80, W - 150, 360, 110
    xmax, ymax = 262144, 140
    X = lambda v: x0 + (x1 - x0) * v / xmax
    Y = lambda v: y0 - (y0 - y1) * v / ymax
    for v in range(0, 141, 20):
        s.line(x0, Y(v), x1, Y(v), t["grid"])
        s.text(x0 - 10, Y(v) + 4, f"{v}", 12, t["faint"], anchor="end")
    for v, lab in [(0, "0"), (65536, "64K"), (131072, "128K"), (196608, "192K"), (262144, "256K")]:
        s.line(X(v), y0, X(v), y0 + 5, t["faint"])
        s.text(X(v), y0 + 22, lab, 12, t["faint"], anchor="middle")
    s.text((x0 + x1) / 2, y0 + 42, "токенов в промпте", 12, t["faint"], anchor="middle")
    s.legend(x0, 92, [(t["base"], "Механика исходного FreeToken"), (t["f1"], "f1"), (t["m128"], "main128")])
    series = (("base", base, t["base"], 3), ("f1", f1, t["f1"], 3), ("m128", m, t["m128"], 3.5))
    for _, vals, colour, w in series:
        pts = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(x_, vals))
        s.add(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{w}" '
              f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, v in zip(x_, vals):
            s.add(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="5" fill="{colour}" stroke="{t["card"]}" stroke-width="2"/>')
    # multiplier of main128 against the original at the right end of the curves
    for i in (3, 4):
        x = x_[i]
        s.text(X(x) - 10, Y(base[i]) - 10, f"{base[i]:g} с", 13, t["muted"], anchor="end")
        s.text(X(x) + 12, (Y(base[i]) + Y(m[i])) / 2 + 5, fmt_x(base[i] / m[i]), 15, t["fg"], weight=700)
    # value strip under the plot
    sy = y0 + 70
    cw = (x1 - x0) / len(x_)
    heads = ["4K", "16K", "64K", "120K", "250K"]
    s.text(24, sy, "Секунды", 12, t["muted"], weight=600)
    for j, h in enumerate(heads):
        s.text(x0 + cw * (j + 0.5), sy, h, 12, t["muted"], anchor="middle", weight=600)
    s.line(24, sy + 8, x1, sy + 8, t["border"])
    for k, (name, vals, colour) in enumerate((("Исходный", base, t["base"]), ("f1", f1, t["f1"]), ("main128", m, t["m128"]))):
        ry = sy + 30 + k * 24
        s.rect(24, ry - 10, 12, 12, colour, rx=3)
        s.text(42, ry, name, 13, t["fg"])
        for j, v in enumerate(vals):
            s.text(x0 + cw * (j + 0.5), ry, f"{v:.2f}" if v < 10 else f"{v:.1f}", 13,
                   t["m128text"] if k == 2 else t["fg"], anchor="middle", weight=600 if k == 2 else None)
    s.text(24, H - 46, "4K–120K: f1 и исходный — матрица 13–14.10; main128 = f1 × отношение main128/f1 16.10 (двух матриц подряд; "
           "64K — отдельный замер, 3 пары).", 11, t["faint"])
    s.text(24, H - 30, "250K: исходная механика — Triton, чанк 4096, замер 13.10; f1 — медиана 11 прогонов; main128 — прямой замер "
           "рычагов main128", 11, t["faint"])
    s.text(24, H - 14, "на профиле long256 (медиана 3). Рост нелинейный: attention квадратичен по длине.", 11, t["faint"])
    return s


def decode_length_chart(t):
    x_, base, f1, m = decode_length_series()
    W, H = 980, 612
    s = Svg(W, H, t, "Скорость генерации по длине свежего промпта",
            "Токенов в секунду. 64–120K — матрица 128K (main128 приведён), 250K — профиль long256. Больше — лучше.")
    x0, x1, y0, y1 = 80, W - 150, 360, 110
    xmax, ymax = 262144, 180
    X = lambda v: x0 + (x1 - x0) * v / xmax
    Y = lambda v: y0 - (y0 - y1) * v / ymax
    for v in range(0, 181, 20):
        s.line(x0, Y(v), x1, Y(v), t["grid"])
        s.text(x0 - 10, Y(v) + 4, f"{v}", 12, t["faint"], anchor="end")
    for v, lab in [(0, "0"), (65536, "64K"), (131072, "128K"), (196608, "192K"), (262144, "256K")]:
        s.line(X(v), y0, X(v), y0 + 5, t["faint"])
        s.text(X(v), y0 + 22, lab, 12, t["faint"], anchor="middle")
    s.text((x0 + x1) / 2, y0 + 42, "токенов в промпте", 12, t["faint"], anchor="middle")
    s.legend(x0, 92, [(t["base"], "Механика исходного FreeToken"), (t["f1"], "f1"), (t["m128"], "main128")])
    series = ((base, t["base"], 3), (f1, t["f1"], 3), (m, t["m128"], 3.5))
    for vals, colour, w in series:
        pts = [(X(x), Y(v)) for x, v in zip(x_, vals) if v is not None]     # points without data are not drawn
        s.add(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{colour}" '
              f'stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>')
        for a, b in pts:
            s.add(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="{colour}" stroke="{t["card"]}" stroke-width="2"/>')
    # multipliers at the key points: main128 against the original, at 250K against f1 (no original there)
    for i in (2, 4):
        s.text(X(x_[i]) + 6, Y(m[i]) - 14, f"+{(m[i] / base[i] - 1) * 100:.0f}%", 15, t["fg"], anchor="middle", weight=700)
    s.text(X(x_[5]) - 14, Y(m[5]) - 14, f"+{(m[5] / f1[5] - 1) * 100:.0f}% к f1", 15, t["fg"], anchor="end", weight=700)
    # value strip under the plot
    sy = y0 + 70
    cw = (x1 - x0) / len(x_)
    heads = ["64", "4K", "16K", "64K", "120K", "250K"]
    s.text(24, sy, "Ток/с", 12, t["muted"], weight=600)
    for j, h in enumerate(heads):
        s.text(x0 + cw * (j + 0.5), sy, h, 12, t["muted"], anchor="middle", weight=600)
    s.line(24, sy + 8, x1, sy + 8, t["border"])
    for k, (name, vals, colour) in enumerate((("Исходный", base, t["base"]), ("f1", f1, t["f1"]), ("main128", m, t["m128"]))):
        ry = sy + 30 + k * 24
        s.rect(24, ry - 10, 12, 12, colour, rx=3)
        s.text(42, ry, name, 13, t["fg"])
        for j, v in enumerate(vals):
            s.text(x0 + cw * (j + 0.5), ry, "—" if v is None else f"{v:.1f}", 13,
                   t["m128text"] if k == 2 else (t["faint"] if v is None else t["fg"]), anchor="middle",
                   weight=600 if k == 2 else None)
    s.text(24, H - 46, "64–120K: f1 и исходный — матрица 13–14.10; main128 = f1 × (main128/f1 из матриц 16.10). "
           "Проценты — main128 к исходному (250K — к f1).", 11, t["faint"])
    s.text(24, H - 30, "250K: у исходной механики замера скорости генерации нет — точка не нарисована; f1 — медиана 11 прогонов; "
           "main128 — прямой замер", 11, t["faint"])
    s.text(24, H - 14, "рычагов main128 на профиле long256 (медиана 3). Скорость падает с длиной: attention читает растущий KV-кэш.",
           11, t["faint"])
    return s


def _fmt_ms(v):
    if v >= 10000:
        return f"{v / 1000:.1f} с"
    if v >= 1000:
        return f"{v / 1000:.2f} с"
    return f"{v:.0f} мс"


def breakdown_chart(t):
    W = 980
    top, rowh = 104, 62
    H = top + rowh * len(SCENARIOS) + 92
    s = Svg(W, H, t, "Куда уходит время: Nsight Systems, релиз main128",
            "Доли окна prefill или шага decode, %. Сумма по строке — 100%. В скобках — то же у f1L0 (настройки f1 на новом коде).")
    x0, x1 = 24, W - 24
    span = x1 - x0
    for i, (req, phase, name) in enumerate(SCENARIOS):
        b, total, _ = nsys_scenario("l9", req, phase)
        _, total0, _ = nsys_scenario("l0", req, phase)
        y = top + i * rowh
        if phase == "prefill":
            when = f"{_fmt_ms(total)} (f1L0: {_fmt_ms(total0)})"
        else:
            when = f"{total:.2f} мс на токен (f1L0: {total0:.2f})"
        s.text(x0, y, name, 14, t["fg"], weight=600)
        s.text(x0 + len(name) * 14 * 0.57 + 10, y, when, 13, t["muted"])
        idle = 100 * b["cpu_gaps_gpu_idle"] / total
        s.text(x1, y, f"простой GPU {idle:.1f}%", 12, t["muted"], anchor="end")
        x = x0
        for bucket, ck, _ in CATS:
            pct = 100 * b.get(bucket, 0.0) / total
            w = span * pct / 100
            if w <= 0:
                continue
            fill = t[ck]
            s.rect(x, y + 10, w, 28, fill)
            if w >= 38:
                s.text(x + w / 2, y + 29, f"{pct:.0f}%", 12, on(t, fill), anchor="middle", weight=600)
            x += w
        s.add(f'<rect x="{x0}" y="{y + 10}" width="{span}" height="28" rx="5" fill="none" stroke="{t["border"]}"/>')
    ly = top + rowh * len(SCENARIOS) + 6
    s.legend(x0, ly, [(t[c], lab) for _, c, lab in CATS[:4]], gap=26)
    s.legend(x0, ly + 24, [(t[c], lab) for _, c, lab in CATS[4:]], gap=26)
    s.text(x0, H - 16, "Источник: results/ladder/nsys.json, релиз main128 (критический путь; перекрытое копирование "
           "в долю не входит). Под профайлером времена завышены: смотрим структуру.", 11, t["faint"])
    return s


def memory_chart(t):
    W, H = 980, 262
    total = sum(v for _, _, v in VRAM)
    s = Svg(W, H, t, f"Видеопамять: {total:.1f} ГиБ на профиле main128",
            "Медиана в soak. Таблица эмбеддингов (0.95 ГиБ) и банк экспертов (18.25 ГБ) лежат в закреплённой ОЗУ; Windows остаётся ≥ 1 ГиБ.")
    x0, x1, y = 24, W - 24, 88
    span = x1 - x0
    x = x0
    for ck, label, v in VRAM:
        w = span * v / total
        fill = t[ck]
        s.rect(x, y, w, 40, fill)
        if w >= 48:
            s.text(x + w / 2, y + 25, f"{v:.2f}", 13, on(t, fill), anchor="middle", weight=600)
        x += w
    s.add(f'<rect x="{x0}" y="{y}" width="{span}" height="40" rx="6" fill="none" stroke="{t["border"]}"/>')
    cols = 3
    for i, (ck, label, v) in enumerate(VRAM):
        cx = x0 + (i % cols) * (span / cols)
        cy = y + 76 + (i // cols) * 26
        s.rect(cx, cy - 11, 12, 12, t[ck], rx=3)
        s.text(cx + 18, cy, f"{label} — {v:.2f} ГиБ", 13, t["muted"])
    return s


def ladder_waterfall(t, kind):
    """Waterfall of the ladder L0..L9 (3 pairs, medians): decode at 16K or TTFT of a 64-token prompt."""
    if kind == "decode":
        vals = ladder_medians("decode_tps", 16384)
        title, sub = "Вклад рычагов: декод на контексте 16K", "Каждая ступень — предыдущая плюс один рычаг, медианы 3 пар, токенов в секунду. Больше — лучше. Ось обрезана."
        lo, hi, ticks, up_good, unit = 130, 180, [130, 140, 150, 160, 170, 180], True, "ток/с"
        fmt, dfmt = (lambda v: f"{v:.1f}"), (lambda d: f"{d:+.1f}".replace("-", "−"))
        rd = lambda v: round(v, 1)
        note = "Красный — ступень не улучшила метрику (CUDA-ядра GDN для prefill на декод не влияют: разница — шум)."
    else:
        vals = ladder_medians("ttft_s", 64)
        title, sub = "Вклад рычагов: TTFT промпта в 64 токена", "Каждая ступень — предыдущая плюс один рычаг, медианы 3 пар, секунды. Меньше — лучше."
        lo, hi, ticks, up_good, unit = 0.0, 0.45, [0.0, 0.1, 0.2, 0.3, 0.4], False, "с"
        fmt, dfmt = (lambda v: f"{v:.3f}"), (lambda d: f"{d:+.3f}".replace("-", "−"))
        rd = lambda v: round(v, 3)
        note = "Красный — ступень не улучшила метрику (изменения до 0.01 с в пределах разброса 3 пар)."
    labels = [lab for _, lab in LADDER_STEPS] + ["main128"]
    n = len(labels)
    W, H = 1000, 520
    s = Svg(W, H, t, title, sub)
    x0, x1, y0, y1 = 70, W - 24, 340, 110
    colw = (x1 - x0) / n
    Y = lambda v: y0 - (y0 - y1) * (v - lo) / (hi - lo)
    for v in ticks:
        s.line(x0, Y(v), x1, Y(v), t["grid"])
        s.text(x0 - 8, Y(v) + 4, (f"{v:g}" if kind == "decode" else f"{v:.1f}"), 12, t["faint"], anchor="end")
    s.text(x0 - 8, y1 - 14, unit, 12, t["faint"], anchor="end")
    bw = colw * 0.56
    prev = None
    for i, lab in enumerate(labels):
        cx = x0 + colw * (i + 0.5)
        if i == 0:
            v = vals[0]
            s.rect(cx - bw / 2, Y(v), bw, Y(lo) - Y(v), t["f1"], rx=3)
            s.text(cx, Y(v) - 8, fmt(v), 13, t["fg"], anchor="middle", weight=700)
            prev = v
        elif i < n - 1:
            v = vals[i]
            d = rd(v) - rd(prev)
            good = (d > 0) == up_good
            colour = t["moe"] if good else t["gather"]
            ya, yb = Y(prev), Y(v)
            s.rect(cx - bw / 2, min(ya, yb), bw, max(abs(ya - yb), 1.5), colour, rx=2)
            s.line(cx - colw + bw / 2, ya, cx - bw / 2, ya, t["faint"], sw=1, dash="3 3")
            s.text(cx, min(ya, yb) - 8, dfmt(d), 12, t["fg"], anchor="middle", weight=600)
            prev = v
        else:
            v = vals[-1]
            s.rect(cx - bw / 2, Y(v), bw, Y(lo) - Y(v), t["m128"], rx=3)
            s.text(cx, Y(v) - 8, fmt(v), 13, t["m128text"], anchor="middle", weight=700)
        words = lab.split()
        lines, cur = [], ""
        for w_ in words:
            if cur and len(cur) + 1 + len(w_) > 11:
                lines.append(cur)
                cur = w_
            else:
                cur = (cur + " " + w_).strip()
        lines.append(cur)
        for j, ln in enumerate(lines):
            s.text(cx, y0 + 20 + j * 14, ln, 11, t["muted"] if 0 < i < n - 1 else t["fg"], anchor="middle")
    foot = ["Источник: results/ladder/summary.json (A/B лесенки, 3 пары); f1 — прошлый выпуск, results/ladder/ab-f1old.json "
            "(другая матрица той же сессии 16.10).",
            "f1L0 — настройки f1 на новом коде: вклад общих правок кода без рычагов, сопоставим с разбросом f1 против собственного алиаса.",
            "Не вошли в релиз: предвыборка экспертов на слой вперёд и прогрев слотов из prefill.",
            note]
    for k, ln in enumerate(foot):
        s.text(24, H - 16 * (len(foot) - k), ln, 11, t["faint"])
    return s


def gpu_floorplan(t, dash, ncu):
    """Three phases, the same GPU block diagram, units coloured by load."""
    W, H = 1000, 660
    s = Svg(W, H, t, "Как загружены блоки GPU изнутри: выпуск f1",
            "RTX 5070 Ti: 70 SM (8960 CUDA-ядер, 280 тензорных), L2 48 МБ, GDDR7 16 ГБ / 896 ГБ/с, PCIe 5.0 x16.")
    sc = dash["scenarios"]
    sp = sc["short_prompt_prefill_278tok"]
    p120 = sc["prefill_120k"]
    d16 = sc["decode_16k"]

    def wavg(pairs):
        tot = sum(w for w, _ in pairs)
        return sum(w * v for w, v in pairs) / tot

    attn_p = ncu_find(ncu, "FlashInfer paged prefill attention, q=2048 chunk, KV 120000")
    moe_gu = ncu_find(ncu, "CUDA NVFP4 MoE expert GEMM gate_up")
    moe_dn = ncu_find(ncu, "CUDA NVFP4 MoE expert GEMM down")
    gdn = [k for k in ncu if k["label"].startswith("GDN ") and not k["label"].startswith("GDN decode")]
    gdn_sm = wavg([(k["duration_us"], k["sm_throughput_pct"]) for k in gdn])
    gdn_dram = wavg([(k["duration_us"], k["dram_throughput_pct"]) for k in gdn])
    moe_sm = wavg([(moe_gu["duration_us"], moe_gu["sm_throughput_pct"]), (moe_dn["duration_us"], moe_dn["sm_throughput_pct"])])
    moe_dram = wavg([(moe_gu["duration_us"], moe_gu["dram_throughput_pct"]), (moe_dn["duration_us"], moe_dn["dram_throughput_pct"])])
    b = {k: v["share_pct"] for k, v in p120["buckets_requested"].items()}
    p_sm = wavg([(b["full_attention"], attn_p["sm_throughput_pct"]), (b["moe_experts"], moe_sm), (b["gdn"], gdn_sm)])
    p_dram = wavg([(b["full_attention"], attn_p["dram_throughput_pct"]), (b["moe_experts"], moe_dram), (b["gdn"], gdn_dram)])
    p_copy = p120["pcie_h2d"]["copy_busy_ms"] / p120["gpu"]["window_ms"] * 100

    attn_d = ncu_find(ncu, "FlashInfer decode attention (tensor-core path), KV 65536")
    fp8a = ncu_find(ncu, "FP8 GEMV QMMA 12288x2048")
    fp8b = ncu_find(ncu, "FP8 GEMV QMMA 9216x2048")
    gdn_d = ncu_find(ncu, "GDN decode")
    with open(os.path.join(NSIGHT, "levers.json"), encoding="utf-8") as f:
        marlin_gbps = json.load(f)["ranked_top3"][2]["evidence"]["marlin_effective_gbps"]
    db = {k: v["share_pct"] for k, v in d16["buckets_requested"].items()}
    fp8_dram = (fp8a["dram_throughput_pct"] + fp8b["dram_throughput_pct"]) / 2
    fp8_sm = (fp8a["sm_throughput_pct"] + fp8b["sm_throughput_pct"]) / 2
    d_dram = wavg([(db["fp8_gemv_gemm"], fp8_dram), (db["moe_experts"], marlin_gbps / 896 * 100),
                   (db["full_attention"], attn_d["dram_throughput_pct"]), (db["gdn"], gdn_d["dram_throughput_pct"])])
    d_sm = wavg([(db["fp8_gemv_gemm"], fp8_sm), (db["full_attention"], attn_d["sm_throughput_pct"]),
                 (db["gdn"], gdn_d["sm_throughput_pct"])])

    sp_copy = sp["pcie_h2d"]["copy_busy_ms"] / sp["gpu"]["window_ms"] * 100
    sp_kern = sp["gpu"]["kernel_busy_pct"]
    sp_exposed = sp["buckets_requested"]["expert_h2d"]["share_pct"]

    panels = [
        ("Короткий промпт", "278 токенов, 0.60 с",
         dict(pcie=sp_copy / 100, pcie_l=f"занят {sp_copy:.0f}% окна, ~35 ГБ/с",
              ce="копирует весь банк, 18.25 ГБ",
              sm=sp_kern / 100, sm_l=f"ядра работают {sp_kern:.0f}% окна",
              l2="MoE на Triton: упор в L1/TEX",
              dram=0.05, dram_l="≈5%",
              verdict=f"Упор: PCIe, {sp_exposed:.0f}% времени ждём копию")),
        ("Prefill 120K", "21.9 с",
         dict(pcie=p_copy / 100, pcie_l=f"занят {p_copy:.1f}% окна, 34 ГБ/с",
              ce="85% копий скрыто за счётом",
              sm=p_sm / 100, sm_l=f"SM ≈{p_sm:.0f}%, тензорные ядра на пределе",
              l2=f"попадания в L2 {attn_p['l2_hit_rate_pct']:.0f}% (attention)",
              dram=p_dram / 100, dram_l=f"≈{p_dram:.0f}%",
              verdict="Упор: тензорные ядра (attention 65%)")),
        ("Decode, контекст 16K", "8.1 мс на токен",
         dict(pcie=db["decode_gather"] / 100, pcie_l=f"{db['decode_gather']:.0f}% шага на ~42 ГБ/с (потолок)",
              ce="TMA gather промахов кэша",
              sm=d_sm / 100, sm_l=f"SM ≈{d_sm:.0f}%: ядра ждут память",
              l2="L2 почти не помогает: веса читаются потоком",
              dram=d_dram / 100, dram_l=f"≈{d_dram:.0f}% пока идут ядра",
              verdict="Упор: видеопамять и PCIe")),
    ]
    pw, gap, px0, py0 = 304, 24, 24, 84
    for i, (name, when, v) in enumerate(panels):
        x = px0 + i * (pw + gap)
        y = py0
        s.rect(x, y, pw, 518, t["panel"], rx=10, stroke=t["border"])
        s.text(x + 16, y + 28, name, 16, t["fg"], weight=700)
        s.text(x + pw - 16, y + 28, when, 12, t["muted"], anchor="end")
        # host RAM
        s.rect(x + 16, y + 44, pw - 32, 40, t["host"], rx=6, stroke=t["border"])
        s.text(x + pw / 2, y + 62, "ОЗУ хоста (pinned)", 12, t["fg"], anchor="middle", weight=600)
        s.text(x + pw / 2, y + 77, "банк экспертов NVFP4, 18.25 ГБ", 11, t["muted"], anchor="middle")
        # PCIe link
        lc = heat(t, v["pcie"])
        s.rect(x + 46, y + 84, 18, 34, lc)
        s.text(x + 74, y + 98, "PCIe 5.0 x16", 11, t["muted"])
        s.text(x + 74, y + 112, v["pcie_l"], 11, t["fg"], weight=600)
        # GPU outline
        gy = y + 124
        s.rect(x + 12, gy, pw - 24, 334, "none", rx=8, stroke=t["faint"], dash="4 3")
        s.text(x + 22, gy + 16, "GPU", 10, t["faint"], weight=700)
        # copy engine
        s.rect(x + 28, gy + 24, pw - 56, 34, lc, rx=6, stroke=t["border"])
        s.text(x + pw / 2, gy + 39, "Copy engine / TMA", 12, on(t, lc), anchor="middle", weight=600)
        s.text(x + pw / 2, gy + 53, v["ce"], 11, on(t, lc), anchor="middle")
        # SM array: 70 tiles, 14 x 5
        sc_ = heat(t, v["sm"])
        tile, tg = 15, 3
        cols, rows = 14, 5
        aw = cols * tile + (cols - 1) * tg
        ax = x + (pw - aw) / 2
        ay = gy + 70
        for r in range(rows):
            for c in range(cols):
                s.rect(ax + c * (tile + tg), ay + r * (tile + tg), tile, tile, sc_, rx=2, stroke=t["border"], sw=0.6)
        s.text(x + pw / 2, ay + rows * (tile + tg) + 14, "70 SM: CUDA-ядра + тензорные ядра", 11, t["muted"], anchor="middle")
        s.text(x + pw / 2, ay + rows * (tile + tg) + 30, v["sm_l"], 12, t["fg"], anchor="middle", weight=600)
        # L2
        l2y = ay + rows * (tile + tg) + 42
        s.rect(x + 28, l2y, pw - 56, 34, t["host"], rx=6, stroke=t["border"])
        s.text(x + pw / 2, l2y + 15, "L2-кэш 48 МБ", 12, t["fg"], anchor="middle", weight=600)
        s.text(x + pw / 2, l2y + 29, v["l2"], 11, t["muted"], anchor="middle")
        # GDDR7: 8 chips of 2 GB on a 256-bit bus
        dc = heat(t, v["dram"])
        my = l2y + 50
        cw = (pw - 56 - 7 * 6) / 8
        for c in range(8):
            s.rect(x + 28 + c * (cw + 6), my, cw, 30, dc, rx=4, stroke=t["border"])
        s.text(x + pw / 2, my + 48, "GDDR7 16 ГБ, 896 ГБ/с", 11, t["muted"], anchor="middle")
        s.text(x + pw / 2, my + 64, f"видеопамять {v['dram_l']}", 12, t["fg"], anchor="middle", weight=600)
        # verdict
        s.rect(x + 12, y + 470, pw - 24, 34, t["card"], rx=6, stroke=t["border"])
        s.text(x + pw / 2, y + 492, v["verdict"], 12, t["fg"], anchor="middle", weight=700)
    # heat legend
    ly = H - 42
    s.text(24, ly + 12, "Загрузка блока:", 12, t["muted"])
    gx = 140
    for k in range(41):
        s.rect(gx + k * 6, ly, 6, 16, heat(t, k / 40))
    s.text(gx - 2, ly + 30, "0%", 10, t["faint"])
    s.text(gx + 246, ly + 30, "100%", 10, t["faint"], anchor="end")
    s.text(400, ly + 6, "Оценка: доли времени из Nsight Systems × загрузка блоков по Nsight Compute", 11, t["faint"])
    s.text(400, ly + 22, "(компоненты с теми же формами). FP8 GEMM в prefill, нормы и marlin ncu не снимались;", 11, t["faint"])
    s.text(400, ly + 38, "для marlin — байты/время из nsys. Это f1; для main128 Nsight Compute не снят (results/ladder/ncu.json).", 11, t["faint"])
    return s


def kernel_heatmap(t, ncu):
    rows = [
        ("Prefill", None),
        ("Attention, FlashInfer", "FlashInfer paged prefill attention, q=2048 chunk, KV 120000", "65% / 19%", "тензорные ядра", None),
        ("MoE NVFP4, наше CUDA-ядро", "CUDA NVFP4 MoE expert GEMM gate_up", "18% / 40%", "тензорные ядра", None),
        ("MoE NVFP4, Triton (было)", "Triton NVFP4 MoE prefill gate_up", "заменено", "кэш L1/TEX", None),
        ("GDN chunk_h, наше", "GDN chunk_h state recurrence", "4% / 8%", "видеопамять", None),
        ("Decode", None),
        ("Attention, FlashInfer", "FlashInfer decode attention (tensor-core path), KV 131072", "7% / 31%", "видеопамять", None),
        ("FP8 GEMV, наше QMMA-ядро", "FP8 GEMV QMMA 12288x2048", "28% / 20%", "видеопамять", None),
        ("MoE decode, marlin", None, "20% / 15%", "видеопамять", "marlin"),
        ("Подкачка экспертов, наш TMA", "decode gather TMA, 4 missing", "18% / 14%", "PCIe", "gather"),
    ]
    cols = [("SM", "sm_throughput_pct"), ("Тензорные ядра¹", "tensor_inst_hmma_pct_peak_active"),
            ("Память, max²", "memory_throughput_pct"), ("GDDR7", "dram_throughput_pct")]
    W = 1000
    rowh, top = 40, 128
    H = top + rowh * len(rows) + 102
    s = Svg(W, H, t, "Загрузка блоков GPU по ядрам: Nsight Compute, выпуск f1",
            "% от пика блока. Доля времени: prefill 120K / 16K и decode 16K / 120K.")
    lx, sx = 24, 252
    mx = 344
    cw = 100
    px = mx + cw * len(cols)
    vx = px + 120
    hy = top - 22
    s.text(sx, hy, "Доля времени", 12, t["muted"], weight=600)
    for j, (name, _) in enumerate(cols):
        s.text(mx + j * cw + cw / 2 - 2, hy, name, 12, t["muted"], anchor="middle", weight=600)
    s.text(px + cw / 2, hy, "PCIe", 12, t["muted"], anchor="middle", weight=600)
    s.text(vx, hy, "Упор в", 12, t["muted"], weight=600)
    s.line(24, top - 12, W - 24, top - 12, t["border"])
    y = top
    for row in rows:
        if row[1] is None and len(row) == 2:
            s.text(lx, y + 24, row[0], 13, t["faint"], weight=700)
            y += rowh
            continue
        label, prefix, share, verdict, special = row
        k = ncu_find(ncu, prefix) if prefix else None
        s.text(lx + 12, y + 24, label, 13, t["fg"], weight=600 if "наш" in label else None)
        s.text(sx, y + 24, share, 13, t["muted"])
        for j, (_, key) in enumerate(cols):
            cx = mx + j * cw
            v = k.get(key) if k else None
            txt, fill = None, t["na"]
            if special == "marlin" and key == "dram_throughput_pct":
                with open(os.path.join(NSIGHT, "levers.json"), encoding="utf-8") as f:
                    g = json.load(f)["ranked_top3"][2]["evidence"]["marlin_effective_gbps"]
                v = g / 896 * 100
                txt = f"≈{v:.0f}%³"
            elif special == "gather" and key not in ("sm_throughput_pct", "dram_throughput_pct"):
                v = None
            if v is not None:
                fill = heat(t, v / 100)
                if txt is None:
                    txt = f"{v:.0f}%" if v >= 1 else "<1%"
                if key == "dram_throughput_pct" and k and k.get("dram_gbps"):
                    txt += f" · {k['dram_gbps']:.0f} ГБ/с"
            else:
                txt = "н/д" if special == "marlin" else "—"
            s.rect(cx + 2, y + 4, cw - 4, rowh - 8, fill, rx=5)
            s.text(cx + cw / 2, y + 25, txt, 12 if "ГБ/с" in txt else 13, on(t, fill) if v is not None else t["faint"],
                   anchor="middle", weight=600 if v is not None else None)
        pc = heat(t, 42 / 45) if special == "gather" else t["na"]
        s.rect(px + 2, y + 4, cw - 4, rowh - 8, pc, rx=5)
        s.text(px + cw / 2, y + 25, "~42 ГБ/с" if special == "gather" else "—", 13,
               on(t, pc) if special == "gather" else t["faint"], anchor="middle", weight=600 if special else None)
        s.text(vx, y + 25, verdict, 13, t["fg"], weight=600)
        y += rowh
    fy = y + 22
    s.text(24, fy, "¹ HMMA, % от пика инструкций; у attention и MoE в prefill главный stall — math_pipe_throttle (конвейер "
           "тензорных ядер занят).", 11, t["faint"])
    s.text(24, fy + 16, "² максимум по L1/TEX, L2 и DRAM.  ³ marlin ncu не снимался: байты/время из nsys.  "
           "PCIe: gather в модели ~42 ГБ/с при потолке канала 41–45 ГБ/с в WSL2.", 11, t["faint"])
    s.text(24, fy + 32, "Ncu снят на компонентах с теми же формами и конфигурацией запуска, что в выпуске. "
           "Источник: results/nsight/ncu-component-kernels.json.", 11, t["faint"])
    s.text(24, fy + 48, "Это выпуск f1. Ядра декода main128 (o-proj GEMV, MoE GEMV) в Nsight Compute не снимались: под ncu сервер не "
           "обрабатывает запросы (results/ladder/ncu.json).", 11, t["faint"])
    return s


def prefill_timeline(t):
    """Per-layer copy vs compute lanes for three prompt sizes (main128, Nsight Systems, mean per layer)."""
    W = 980
    blocks = []
    for req, title in (("warm_decode", "Короткий промпт, 278 токенов: путь по чанкам"),
                       ("fresh_16k", "Свежий промпт 16K: послойный prefill"),
                       ("fresh_120k", "Свежий промпт 120K: послойный prefill")):
        _, total, tr = nsys_scenario("l9", req, "prefill")
        copy = tr["h2d"]["union_ms"] / 40
        period = total / 40
        comp = tr["kernel_union_ms"] / 40
        blocks.append((title, copy, period, comp, tr["h2d"]["overlap_percent"]))
    bh = 108
    H = 104 + bh * len(blocks) + 40
    s = Svg(W, H, t, "Копирование экспертов и вычисления, слой за слоем (main128)",
            "Средние длительности на слой из таймлайна Nsight Systems, первые 3 из 40 слоёв, каждая строка в своём масштабе.")
    s.legend(24, 92, [(t["h2d"], "PCIe: эксперты слоя (~456 МБ)"), (t["moe"], "GPU: вычисления слоя")])
    x0, x1 = 150, W - 40
    for i, (name, copy, period, comp, ov) in enumerate(blocks):
        y = 124 + i * bh
        s.text(24, y, name, 14, t["fg"], weight=600)
        s.text(W - 24, y, f"скрыто за вычислениями: {ov:.0f}% копий", 12, t["muted"], anchor="end")
        layers = 3
        # timeline: layer L copy then compute; next layer's copy starts as soon as the link is free
        events = []
        if ov < 50:
            # chunked path: copy, then compute; little overlap
            tcur = 0.0
            for L in range(layers):
                events.append(("copy", L, tcur, tcur + copy))
                events.append(("comp", L, tcur + copy, tcur + copy + comp))
                tcur += period
        else:
            events.append(("copy", 0, 0.0, copy))
            tc = copy
            for L in range(layers):
                events.append(("comp", L, tc, tc + comp))
                if L + 1 < layers:
                    events.append(("copy", L + 1, tc, tc + copy))
                tc += max(comp, copy)
        tmax = max(e[3] for e in events)
        X = lambda v: x0 + (x1 - x0) * v / tmax
        for lane, (lab, kind) in enumerate((("PCIe", "copy"), ("GPU", "comp"))):
            ly = y + 14 + lane * 30
            s.text(x0 - 12, ly + 17, lab, 12, t["muted"], anchor="end")
            s.rect(x0, ly, x1 - x0, 24, t["panel"], rx=4, stroke=t["border"], sw=0.6)
            for kind2, L, a, b in events:
                if kind2 != kind:
                    continue
                fill = t["h2d"] if kind == "copy" else t["moe"]
                s.rect(X(a), ly + 2, max(X(b) - X(a), 2), 20, fill, rx=3)
                if X(b) - X(a) > 70:
                    s.text((X(a) + X(b)) / 2, ly + 16, f"слой {L + 1} · {b - a:.0f} мс" if b - a >= 1 else f"слой {L + 1}",
                           11, on(t, fill), anchor="middle", weight=600)
                elif X(b) - X(a) > 24:
                    s.text((X(a) + X(b)) / 2, ly + 16, f"{L + 1}", 11, on(t, fill), anchor="middle", weight=600)
        s.text(x1, y + 88, f"0 … {tmax:.0f} мс", 11, t["faint"], anchor="end")
    s.text(24, H - 16, "Короткий промпт ждёт PCIe: копия слоя (~8 мс) дольше его счёта (~3 мс). Куски короче 160 токенов везут только выбранных "
           "роутером экспертов; уже лежащие в VRAM эксперты копируются внутри GPU.", 11, t["faint"])
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    dash, ncu = load_nsight()
    written = []
    for theme, t in THEMES.items():
        written.append(paired_bars(t, "ttft").save("ttft-128k", theme))
        written.append(paired_bars(t, "decode").save("decode-128k", theme))
        written.append(length_chart(t).save("ttft-length", theme))
        written.append(decode_length_chart(t).save("decode-length", theme))
        written.append(breakdown_chart(t).save("time-breakdown", theme))
        written.append(memory_chart(t).save("vram", theme))
        written.append(gpu_floorplan(t, dash, ncu).save("gpu-load", theme))
        written.append(kernel_heatmap(t, ncu).save("gpu-kernels", theme))
        written.append(prefill_timeline(t).save("prefill-timeline", theme))
        written.append(ladder_waterfall(t, "decode").save("ladder-decode16k", theme))
        written.append(ladder_waterfall(t, "ttft").save("ladder-ttft64", theme))
    for p in written:
        print(os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
