#!/usr/bin/env python3
"""Generate the README charts as light and dark SVG files in docs/img/.

No third-party packages. Nsight numbers are read from results/nsight/*.json;
matrix, long-context and memory numbers are copied from the README tables
(sources noted next to each block).

    python3 tools/readme_charts.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")
NSIGHT = os.path.join(ROOT, "results", "nsight")

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"

THEMES = {
    "light": {
        "bg": "#ffffff", "card": "#f6f8fa", "border": "#d0d7de", "fg": "#1f2328",
        "muted": "#59636e", "faint": "#8c959f", "grid": "#e1e6eb", "panel": "#ffffff",
        "base": "#a8b1ba", "f1": "#4d8c00", "f1text": "#3d7000",
        "attn": "#8250df", "moe": "#1a7f37", "fp8": "#0969da", "gdn": "#1b7c83",
        "norms": "#bf3989", "h2d": "#d1561a", "gather": "#cf222e", "idle": "#d0d7de",
        "host": "#eaeef2", "heat": ["#f2f5f8", "#fde9b6", "#f9b46a", "#ec6a2c", "#c5221f"],
        "na": "#eaeef2",
    },
    "dark": {
        "bg": "#0d1117", "card": "#161b22", "border": "#30363d", "fg": "#e6edf3",
        "muted": "#9198a1", "faint": "#6e7681", "grid": "#262c34", "panel": "#0d1117",
        "base": "#4f5864", "f1": "#76b900", "f1text": "#8fd400",
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

# results/matrix/final-f1-m-main-release-vs-bf128-comparison.json (medians, README §1)
MATRIX = [
    # label lines, TTFT bf128, TTFT f1, TTFT ratio label, decode bf128, decode f1
    (["Промпт 64 токена"], 0.75, 0.79, "×0.94", 96.4, 122.8),
    (["Свежий промпт 4K"], 2.24, 0.77, "×2.9", 98.9, 124.5),
    (["Свежий промпт 16K"], 6.78, 1.59, "×4.3", 94.2, 119.1),
    (["Свежий промпт 64K"], 28.46, 9.14, "×3.1", 84.4, 99.3),
    (["Свежий промпт 120K"], 54.77, 22.07, "×2.5", 80.7, 94.6),
    (["Продолжение диалога, 4K"], 2.20, 0.78, "×2.9", 106.4, 126.9),
    (["Правка последнего", "сообщения, 16K"], 6.85, 0.78, "×8.8", 90.4, 112.4),
    (["Правка результата", "инструмента, 16K"], 7.11, 0.76, "×9.4", 94.9, 113.4),
]

# README §1 (matrix) for 4K-120K; README §3 for 250K (f1: long256, median of 11;
# base: original FreeToken mechanics, Triton, chunk 4096, measured 13.10).
LENGTH = {
    "x": [4096, 16384, 65536, 120000, 250000],
    "base": [2.24, 6.78, 28.46, 54.77, 126.5],
    "f1": [0.77, 1.59, 9.14, 22.07, 77.4],
}

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

SCENARIOS = [
    ("short_prompt_prefill_278tok", "Короткий промпт, 278 токенов", "prefill 0.60 с"),
    ("prefill_16k", "Prefill 16K", "1.53 с"),
    ("prefill_120k", "Prefill 120K", "21.9 с"),
    ("decode_16k", "Decode на контексте 16K", "8.1 мс на токен"),
    ("decode_120k", "Decode на контексте 120K", "11.4 мс на токен"),
]

# README «Память и KV-кэш» (soak median, main128)
VRAM = [
    ("moe", "Кэш экспертов, 3164 слота", 5.25),
    ("fp8", "Плотные веса и масштабы", 2.52),
    ("attn", "KV-кэш BF16, 131K токенов", 2.50),
    ("gdn", "Состояния GDN", 0.54),
    ("norms", "Прочее PyTorch", 0.60),
    ("base", "CUDA, cuBLAS, рабочий стол", 2.90),
    ("idle", "Свободно (медиана)", 1.63),
]


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

def paired_bars(t, kind):
    """Horizontal paired bars: original FreeToken vs f1 (TTFT or decode)."""
    W, rowh, top = 980, 50, 112
    H = top + rowh * len(MATRIX) + 58
    if kind == "ttft":
        title = "Время до первого токена: исходный FreeToken против f1"
        sub = "Парная матрица, контекст 128K, медианы 5 раундов, секунды. Меньше — лучше."
        vmax, ticks, unit = 60, [0, 10, 20, 30, 40, 50, 60], "с"
    else:
        title = "Скорость генерации: исходный FreeToken против f1"
        sub = "Та же матрица, ответ 128 токенов, токенов в секунду. Больше — лучше."
        vmax, ticks, unit = 140, [0, 20, 40, 60, 80, 100, 120, 140], "ток/с"
    s = Svg(W, H, t, title, sub)
    x0, x1 = 250, W - 120
    span = x1 - x0
    s.legend(24, 92, [(t["base"], "Исходный FreeToken (bf128)"), (t["f1"], "Выпуск f1")])
    s.text(W - 24, 92, "f1 против исходного", 12, t["muted"], anchor="end")
    for v in ticks:
        x = x0 + span * v / vmax
        s.line(x, top - 6, x, top + rowh * len(MATRIX), t["grid"])
        s.text(x, top + rowh * len(MATRIX) + 18, f"{v}", 12, t["faint"], anchor="middle")
    s.text(x1, top + rowh * len(MATRIX) + 38, unit, 12, t["faint"], anchor="end")
    for i, (label, tb, tf, tr, db, df) in enumerate(MATRIX):
        y = top + i * rowh
        if i % 2 == 1:
            s.rect(12, y - 4, W - 24, rowh, t["panel"], rx=6, opacity=0.55)
        base, f1 = (tb, tf) if kind == "ttft" else (db, df)
        ly = y + rowh / 2 - (len(label) - 1) * 8 + 4
        for j, part in enumerate(label):
            s.text(24, ly + j * 16, part, 14, t["fg"])
        for k, (v, colour) in enumerate(((base, t["base"]), (f1, t["f1"]))):
            by = y + 7 + k * 19
            w = span * v / vmax
            s.rect(x0, by, w, 14, colour, rx=3)
            txt = f"{v:.2f}" if kind == "ttft" else f"{v:.1f}"
            s.text(x0 + w + 6, by + 11.5, txt, 12, t["fg"] if k else t["muted"], weight=600 if k else None)
        if kind == "ttft":
            badge = tr
            good = tf < tb
        else:
            badge = f"+{(df / db - 1) * 100:.0f}%"
            good = True
        s.text(W - 24, y + rowh / 2 + 5, badge, 17, t["f1text"] if good else t["muted"], anchor="end", weight=700)
        if kind == "ttft" and not good:
            s.text(W - 24, y + rowh / 2 + 19, "вне гейта", 10, t["faint"], anchor="end")
    return s


def length_chart(t):
    W, H = 980, 490
    s = Svg(W, H, t, "TTFT по длине свежего промпта",
            "Секунды до первого токена. До 120K — парная матрица main128, 250K — профиль long256.")
    x0, x1, y0, y1 = 80, W - 150, 400, 110
    xmax, ymax = 262144, 140
    X = lambda v: x0 + (x1 - x0) * v / xmax
    Y = lambda v: y0 - (y0 - y1) * v / ymax
    for v in range(0, 141, 20):
        s.line(x0, Y(v), x1, Y(v), t["grid"])
        s.text(x0 - 10, Y(v) + 4, f"{v}", 12, t["faint"], anchor="end")
    for v, lab in [(0, "0"), (65536, "64K"), (131072, "128K"), (196608, "192K"), (262144, "256K")]:
        s.line(X(v), y0, X(v), y0 + 5, t["faint"])
        s.text(X(v), y0 + 22, lab, 12, t["faint"], anchor="middle")
    s.text((x0 + x1) / 2, y0 + 44, "токенов в промпте", 12, t["faint"], anchor="middle")
    s.legend(x0, 92, [(t["base"], "Механика исходного FreeToken"), (t["f1"], "Выпуск f1")])
    for key, colour, w in (("base", t["base"], 3), ("f1", t["f1"], 3.5)):
        pts = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(LENGTH["x"], LENGTH[key]))
        s.add(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{w}" '
              f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, v in zip(LENGTH["x"], LENGTH[key]):
            s.add(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="5" fill="{colour}" stroke="{t["card"]}" stroke-width="2"/>')
    for i in (2, 3, 4):
        x = LENGTH["x"][i]
        b, f = LENGTH["base"][i], LENGTH["f1"][i]
        s.text(X(x) - 10, Y(b) - 10, f"{b:g} с", 13, t["muted"], anchor="end")
        s.text(X(x) + 10, Y(f) + 18, f"{f:g} с", 13, t["f1text"], weight=600)
        s.text(X(x) + 10, (Y(b) + Y(f)) / 2 + 5, f"×{b / f:.1f}", 15, t["fg"], weight=700)
    s.text(x0, H - 14, "250K: f1 — медиана 11 прогонов; исходная механика — Triton, чанк 4096, замер 13.10. "
           "Рост нелинейный: attention квадратичен по длине.", 11, t["faint"])
    return s


def breakdown_chart(t, dash):
    W = 980
    top, rowh = 104, 62
    H = top + rowh * len(SCENARIOS) + 92
    s = Svg(W, H, t, "Куда уходит время: Nsight Systems по выпускной сборке",
            "Доли окна prefill или шага decode, %. Сумма по строке — 100%.")
    x0, x1 = 24, W - 24
    span = x1 - x0
    for i, (key, name, when) in enumerate(SCENARIOS):
        sc = dash["scenarios"][key]
        b = sc["buckets_requested"]
        y = top + i * rowh
        s.text(x0, y, name, 14, t["fg"], weight=600)
        s.text(x0 + len(name) * 14 * 0.57 + 10, y, when, 13, t["muted"])
        busy = sc["gpu"]["busy_pct"]
        s.text(x1, y, f"GPU занят {busy:.1f}%", 12, t["muted"], anchor="end")
        x = x0
        for bucket, ck, _ in CATS:
            pct = b[bucket]["share_pct"]
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
    s.text(x0, H - 16, "Источник: results/nsight/dashboard.json (критический путь; перекрытое копирование "
           "в долю не входит, длительности не суммируются).", 11, t["faint"])
    return s


def memory_chart(t):
    W, H = 980, 250
    total = sum(v for _, _, v in VRAM)
    s = Svg(W, H, t, f"Видеопамять: {total:.1f} ГиБ на профиле main128",
            "Медиана в soak. Эксперты, которые не влезли в слоты, лежат в закреплённой ОЗУ (банк 18.25 ГБ).")
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


def gpu_floorplan(t, dash, ncu):
    """Three phases, the same GPU block diagram, units coloured by load."""
    W, H = 1000, 660
    s = Svg(W, H, t, "Как загружены блоки GPU изнутри",
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
    s.text(400, ly + 38, "для marlin — байты/время из nsys.", 11, t["faint"])
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
    H = top + rowh * len(rows) + 86
    s = Svg(W, H, t, "Загрузка блоков GPU по ядрам: Nsight Compute",
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
    return s


def prefill_timeline(t, dash):
    """Per-layer copy vs compute lanes for three prompt sizes (mean durations)."""
    sc = dash["scenarios"]
    W = 980
    blocks = []
    for key, name in (("short_prompt_prefill_278tok", "Короткий промпт, 278 токенов: путь по чанкам"),
                      ("prefill_16k", "Свежий промпт 16K: послойный prefill"),
                      ("prefill_120k", "Свежий промпт 120K: послойный prefill")):
        s_ = sc[key]
        copy = s_["pcie_h2d"]["copy_busy_ms"] / 40
        period = s_["gpu"]["window_ms"] / 40
        comp = s_["gpu"]["window_ms"] * s_["gpu"]["kernel_busy_pct"] / 100 / 40
        blocks.append((name, copy, period, comp, s_["pcie_h2d"]["copy_overlapped_with_kernels_pct"]))
    bh = 108
    H = 104 + bh * len(blocks) + 40
    s = Svg(W, H, t, "Копирование экспертов и вычисления, слой за слоем",
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
    s.text(24, H - 16, "Короткие промпты ждут PCIe: копия слоя (~13 мс) дольше его счёта. "
           "В следующем обновлении эксперты из VRAM будут копироваться внутри GPU (hit-d2d).", 11, t["faint"])
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    dash, ncu = load_nsight()
    written = []
    for theme, t in THEMES.items():
        written.append(paired_bars(t, "ttft").save("ttft-128k", theme))
        written.append(paired_bars(t, "decode").save("decode-128k", theme))
        written.append(length_chart(t).save("ttft-length", theme))
        written.append(breakdown_chart(t, dash).save("time-breakdown", theme))
        written.append(memory_chart(t).save("vram", theme))
        written.append(gpu_floorplan(t, dash, ncu).save("gpu-load", theme))
        written.append(kernel_heatmap(t, ncu).save("gpu-kernels", theme))
        written.append(prefill_timeline(t, dash).save("prefill-timeline", theme))
    for p in written:
        print(os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
