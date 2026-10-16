#!/usr/bin/env python3
"""Low-level schemes of the main128 decode kernels and prefill transport ("how it works inside").

One scene description per figure, three outputs: docs/img/kernel-<name>.svg and kernel-<name>-dark.svg
(README, light and dark) and docs/kernels-lowlevel.drawio (one draw.io page per figure, editable).
Every number and every name on a scheme is taken from the code or from results/ladder/*.json; the
source file of each scheme is listed in its SOURCES entry and printed with --sources.
Stdlib only, deterministic. Usage: python tools/kernels-diagram.py [--sources]
"""
import html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = '\n'

PALETTE = {
    'light': dict(bg='#f6f7f9', fg='#17202b', muted='#5b6675', line='#7d8a99', group='#ffffff', groupline='#d3d9e1',
                  op='#ffffff', opline='#9aa6b4', lever='#e3f0f5', leverline='#1f6f8b', mem='#f1f3f6', memline='#b5bfcb',
                  memhot='#e5f3ea', memhotline='#1d7a46', io='#17202b', iofg='#ffffff', link='#fff5e0', linkline='#b7791f',
                  old='#fbe9e7', oldline='#b3402f', sync='#6b46c1',
                  z0='#dcebfa', z0line='#2b6cb0', z1='#e1f3e8', z1line='#1d7a46', z2='#fdebd0', z2line='#b7791f',
                  z3='#efe3f7', z3line='#7b3fa0'),
    'dark': dict(bg='#0f141a', fg='#e4e9ef', muted='#97a3b2', line='#7d8a99', group='#151c24', groupline='#2a3440',
                 op='#1b232d', opline='#5c6978', lever='#16303b', leverline='#5fb3cf', mem='#1b232d', memline='#3a4654',
                 memhot='#17311f', memhotline='#5cc489', io='#e4e9ef', iofg='#0f141a', link='#33290f', linkline='#e0a84a',
                 old='#3a1f1b', oldline='#f08a7a', sync='#b794f4',
                 z0='#172a3f', z0line='#6aaef0', z1='#17311f', z1line='#5cc489', z2='#3a2c12', z2line='#e0a84a',
                 z3='#2c1f3a', z3line='#c08bea'),
}
FILL = {  # kind -> (fill, stroke, text, stroke width)
    'op': ('op', 'opline', 'fg', 1.2), 'lever': ('lever', 'leverline', 'fg', 2), 'old': ('old', 'oldline', 'fg', 1.6),
    'mem': ('mem', 'memline', 'fg', 1.2), 'memhot': ('memhot', 'memhotline', 'fg', 2), 'io': ('io', 'io', 'iofg', 1.2),
    'link': ('link', 'linkline', 'fg', 2), 'z0': ('z0', 'z0line', 'fg', 1.6), 'z1': ('z1', 'z1line', 'fg', 1.6),
    'z2': ('z2', 'z2line', 'fg', 1.6), 'z3': ('z3', 'z3line', 'fg', 1.6),
}
COLORS = {'fg': 'fg', 'muted': 'muted', 'lever': 'leverline', 'warn': 'linkline', 'old': 'oldline', 'sync': 'sync',
          'good': 'memhotline', 'z0': 'z0line', 'z1': 'z1line', 'z2': 'z2line', 'z3': 'z3line'}
ARROWS = {'flow': ('line', None), 'xfer': ('linkline', '7 4'), 'sync': ('sync', '2 4'), 'lever': ('leverline', None),
          'old': ('oldline', None), 'a0': ('z0line', None), 'a1': ('z1line', None), 'a2': ('z2line', None)}

WARNINGS = []


# ---------------------------------------------------------------------------------------------
# text metrics (a conservative estimate; the real widths are checked in a browser)
# ---------------------------------------------------------------------------------------------
def char_w(ch):
    if ch in '  ':
        return 0.30
    if ch in "iıl.,:;|!'’·":
        return 0.28
    if ch in '()[]{}/\\-–−':
        return 0.36
    if ch in 'frtjгтлпиcsaeozувнкхзэчьъ':
        return 0.52
    if ch in 'mwMWшщыюжМШЩЫЮ—…':
        return 0.86
    if ch.isupper() or ch in 'ФДБ':
        return 0.66
    if ch.isdigit():
        return 0.56
    return 0.58


def text_w(s, size, weight=400):
    return sum(char_w(c) for c in s) * size * (1.07 if weight >= 600 else 1.0) * 1.04


def wrap(text, width, size, weight=400):
    out = []
    for para in text.split('\n'):
        para = re.sub(r'\[[^\]]*\]', lambda m: m.group(0).replace(' ', '\u00a0'), para)
        cur = ''
        for word in para.split(' '):
            trial = f'{cur} {word}'.strip()
            if cur and text_w(trial, size, weight) > width:
                out.append(cur)
                cur = word
            else:
                cur = trial
        out.append(cur)
    return out


# ---------------------------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------------------------
class Scene:
    def __init__(self, key, title, subtitle, w, h, sources):
        self.key, self.title, self.subtitle, self.w, self.h, self.sources = key, title, subtitle, w, h, sources
        self.items = []
        self.n = 0

    def _id(self, prefix):
        self.n += 1
        return f'{self.key}-{prefix}{self.n}'

    def group(self, x, y, w, h, title='', tcolor='muted'):
        self.items.append(dict(t='group', id=self._id('g'), x=x, y=y, w=w, h=h, title=title, tcolor=tcolor))

    def box(self, x, y, w, h, kind, title, detail='', tsize=12.5, dsize=11, align='center'):
        lines = [(s, tsize, 600, 'fg') for s in wrap(title, w - 14, tsize, 600)] if title else []
        if detail:
            lines += [(s, dsize, 400, 'muted') for s in wrap(detail, w - 14, dsize)]
        need = sum(sz * 1.3 for _, sz, _, _ in lines) + 8
        if need > h:
            WARNINGS.append(f'{self.key}: box "{title[:40]}" needs {need:.0f}px, has {h}')
        self.items.append(dict(t='box', id=self._id('b'), x=x, y=y, w=w, h=h, kind=kind, title=title, detail=detail,
                               lines=lines, align=align, tsize=tsize, dsize=dsize))

    def ghost(self, x, y, w, h, title, kind='old', size=11):
        """dashed outline, no fill: a wait or an idle gap"""
        self.items.append(dict(t='ghost', id=self._id('x'), x=x, y=y, w=w, h=h, kind=kind, title=title, size=size,
                               lines=wrap(title, w - 8, size)))

    def text(self, x, y, s, size=12, weight=400, color='muted', anchor='start'):
        self.items.append(dict(t='text', id=self._id('t'), x=x, y=y, s=s, size=size, weight=weight, color=color,
                               anchor=anchor))

    def note(self, x, y, w, s, size=11.5, color='muted', weight=400, anchor='start', lh=1.35):
        """wrapped free text; (x, y) is the top-left corner"""
        lines = wrap(s, w, size, weight)
        for i, line in enumerate(lines):
            tx = x if anchor == 'start' else x + w / 2 if anchor == 'middle' else x + w
            self.text(tx, y + size + i * size * lh, line, size, weight, color, anchor)
        return y + len(lines) * size * lh + 2

    def arrow(self, pts, kind='flow', label='', lx=None, ly=None, width=1.7, head=True):
        self.items.append(dict(t='arrow', id=self._id('a'), pts=pts, kind=kind, label=label, lx=lx, ly=ly, width=width,
                               head=head))

    def line(self, x1, y1, x2, y2, kind='flow', width=1.2):
        self.arrow([(x1, y1), (x2, y2)], kind, width=width, head=False)

    def tile(self, x, y, w, h, kind, label='', size=10):
        self.items.append(dict(t='tile', id=self._id('c'), x=x, y=y, w=w, h=h, kind=kind, label=label, size=size))


# ---------------------------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------------------------
def esc(s):
    return html.escape(s, quote=False)


def render_svg(sc, theme):
    p = PALETTE[theme]
    W, H = sc.w, sc.h
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="IBM Plex Sans, Segoe UI, Arial, sans-serif" role="img" aria-label="{html.escape(sc.title)}">',
         f'<title>{esc(sc.title)}</title>', f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>', '<defs>']
    for name, (col, _) in ARROWS.items():
        o.append(f'<marker id="m-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                 f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{p[col]}"/></marker>')
    o.append('</defs>')
    o.append(f'<text x="20" y="32" font-size="20" font-weight="600" fill="{p["fg"]}">{esc(sc.title)}</text>')
    if sc.subtitle:
        for i, line in enumerate(wrap(sc.subtitle, W - 40, 12.5)):
            o.append(f'<text x="20" y="{54 + i * 17}" font-size="12.5" fill="{p["muted"]}">{esc(line)}</text>')
    for it in sc.items:
        if it['t'] == 'group':
            o.append(f'<rect x="{it["x"]}" y="{it["y"]}" width="{it["w"]}" height="{it["h"]}" rx="8" fill="{p["group"]}" '
                     f'stroke="{p["groupline"]}" stroke-dasharray="5 4"/>')
            if it['title']:
                o.append(f'<text x="{it["x"] + 14}" y="{it["y"] + 22}" font-size="13.5" font-weight="600" '
                         f'fill="{p[COLORS[it["tcolor"]]]}">{esc(it["title"])}</text>')
    for it in sc.items:
        t = it['t']
        if t in ('box', 'tile'):
            fill, stroke, fg, sw = FILL[it['kind']]
            rx = 6 if t == 'box' else 3
            o.append(f'<rect x="{it["x"]}" y="{it["y"]}" width="{it["w"]}" height="{it["h"]}" rx="{rx}" '
                         f'fill="{p[fill]}" stroke="{p[stroke]}" stroke-width="{sw}"/>')
            if t == 'tile':
                if it['label']:
                    o.append(f'<text x="{it["x"] + it["w"] / 2}" y="{it["y"] + it["h"] / 2 + it["size"] * 0.35}" '
                             f'font-size="{it["size"]}" fill="{p[fg]}" text-anchor="middle">{esc(it["label"])}</text>')
                continue
            lines = it['lines']
            total = sum(sz * 1.3 for _, sz, _, _ in lines)
            y = it['y'] + it['h'] / 2 - total / 2
            for s, sz, wt, col in lines:
                y += sz * 1.3
                color = p[fg] if col == 'fg' else (p['iofg'] if fg == 'iofg' else p['muted'])
                tx = it['x'] + it['w'] / 2 if it['align'] == 'center' else it['x'] + 8
                anchor = 'middle' if it['align'] == 'center' else 'start'
                o.append(f'<text x="{tx}" y="{y - sz * 0.3:.1f}" font-size="{sz}" font-weight="{wt}" fill="{color}" '
                         f'text-anchor="{anchor}">{esc(s)}</text>')
        elif t == 'ghost':
            col = p[FILL[it['kind']][1]]
            o.append(f'<rect x="{it["x"]}" y="{it["y"]}" width="{it["w"]}" height="{it["h"]}" rx="5" fill="none" '
                     f'stroke="{col}" stroke-width="1.4" stroke-dasharray="4 3"/>')
            total = len(it['lines']) * it['size'] * 1.3
            y = it['y'] + it['h'] / 2 - total / 2
            for s in it['lines']:
                y += it['size'] * 1.3
                o.append(f'<text x="{it["x"] + it["w"] / 2}" y="{y - it["size"] * 0.3:.1f}" font-size="{it["size"]}" '
                         f'fill="{col}" text-anchor="middle">{esc(s)}</text>')
        elif t == 'text':
            o.append(f'<text x="{it["x"]}" y="{it["y"]}" font-size="{it["size"]}" font-weight="{it["weight"]}" '
                     f'fill="{p[COLORS[it["color"]]]}" text-anchor="{it["anchor"]}">{esc(it["s"])}</text>')
    for it in sc.items:
        if it['t'] != 'arrow':
            continue
        col, dash = ARROWS[it['kind']]
        pts = ' '.join(f'{x},{y}' for x, y in it['pts'])
        extra = (f' stroke-dasharray="{dash}"' if dash else '') + (f' marker-end="url(#m-{it["kind"]})"' if it['head'] else '')
        o.append(f'<polyline points="{pts}" fill="none" stroke="{p[col]}" stroke-width="{it["width"]}"{extra}/>')
        if it['label']:
            (x1, y1), (x2, y2) = it['pts'][0], it['pts'][-1]
            lx = it['lx'] if it['lx'] is not None else (x1 + x2) / 2
            ly = it['ly'] if it['ly'] is not None else (y1 + y2) / 2 - 5
            tw = text_w(it['label'], 11) + 8
            o.append(f'<rect x="{lx - tw / 2:.1f}" y="{ly - 11:.1f}" width="{tw:.1f}" height="15" rx="3" fill="{p["bg"]}" '
                     f'fill-opacity="0.9"/>')
            o.append(f'<text x="{lx}" y="{ly}" font-size="11" fill="{p[col]}" text-anchor="middle">{esc(it["label"])}</text>')
    o.append('</svg>')
    return NL.join(o) + NL


# ---------------------------------------------------------------------------------------------
# draw.io
# ---------------------------------------------------------------------------------------------
def drawio_page(sc):
    p = PALETTE['light']
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']

    def vertex(i, x, y, w, h, value, style):
        cells.append(f'<mxCell id="{i}" value="{html.escape(value, quote=True)}" style="{style}html=1;" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    vertex(f'{sc.key}-title', 20, 8, sc.w - 40, 30, f'<b>{html.escape(sc.title)}</b>',
           f'text;fontSize=18;align=left;verticalAlign=middle;fontColor={p["fg"]};')
    if sc.subtitle:
        vertex(f'{sc.key}-sub', 20, 40, sc.w - 40, 34, html.escape(sc.subtitle),
               f'text;fontSize=11;align=left;verticalAlign=top;whiteSpace=wrap;fontColor={p["muted"]};')
    order = sorted(sc.items, key=lambda it: it['t'] != 'group')
    for it in order:
        t = it['t']
        if t == 'group':
            vertex(it['id'], it['x'], it['y'], it['w'], it['h'], f'<b>{html.escape(it["title"])}</b>',
                   f'rounded=1;dashed=1;fillColor={p["group"]};strokeColor={p["groupline"]};verticalAlign=top;align=left;'
                   f'spacingLeft=10;fontStyle=1;fontSize=12;fontColor={p[COLORS[it["tcolor"]]]};')
        elif t == 'box':
            fill, stroke, fg, sw = FILL[it['kind']]
            value = f'<b>{html.escape(it["title"])}</b>' if it['title'] else ''
            if it['detail']:
                value += ('<br>' if value else '') + f'<font style="font-size:{int(it["dsize"])}px">{html.escape(it["detail"]).replace(chr(10), "<br>")}</font>'
            vertex(it['id'], it['x'], it['y'], it['w'], it['h'], value,
                   f'rounded=1;whiteSpace=wrap;fontSize={int(it["tsize"])};fillColor={p[fill]};strokeColor={p[stroke]};'
                   f'strokeWidth={sw};fontColor={p[fg]};' + ('align=left;spacingLeft=6;' if it['align'] != 'center' else ''))
        elif t == 'tile':
            fill, stroke, fg, sw = FILL[it['kind']]
            vertex(it['id'], it['x'], it['y'], it['w'], it['h'], html.escape(it['label']),
                   f'rounded=1;whiteSpace=wrap;fontSize={int(it["size"])};fillColor={p[fill]};strokeColor={p[stroke]};'
                   f'strokeWidth={sw};fontColor={p[fg]};spacing=0;')
        elif t == 'ghost':
            stroke = FILL[it['kind']][1]
            vertex(it['id'], it['x'], it['y'], it['w'], it['h'], html.escape(it['title']),
                   f'rounded=1;dashed=1;whiteSpace=wrap;fontSize={int(it["size"])};fillColor=none;strokeColor={p[stroke]};'
                   f'fontColor={p[stroke]};')
        elif t == 'text':
            align = {'start': 'left', 'middle': 'center', 'end': 'right'}[it['anchor']]
            w = max(40, int(text_w(it['s'], it['size'], it['weight'])) + 10)
            x = it['x'] if it['anchor'] == 'start' else it['x'] - w / 2 if it['anchor'] == 'middle' else it['x'] - w
            vertex(it['id'], round(x, 1), round(it['y'] - it['size'] - 2, 1), w, round(it['size'] + 8, 1), html.escape(it['s']),
                   f'text;fontSize={int(it["size"])};align={align};verticalAlign=middle;spacing=0;'
                   f'fontStyle={1 if it["weight"] >= 600 else 0};fontColor={p[COLORS[it["color"]]]};')
    for it in sc.items:
        if it['t'] != 'arrow':
            continue
        col, dash = ARROWS[it['kind']]
        pts = it['pts']
        mid = ''.join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
        arr = '<Array as="points">' + mid + '</Array>' if mid else ''
        style = (f'endArrow={"block" if it["head"] else "none"};html=1;rounded=0;strokeColor={p[col]};strokeWidth={it["width"]};'
                 + ('dashed=1;' if it['kind'] == 'xfer' else 'dashed=1;dashPattern=2 4;' if it['kind'] == 'sync' else ''))
        cells.append(f'<mxCell id="{it["id"]}" value="{html.escape(it["label"], quote=True)}" style="{style}fontSize=10;'
                     f'fontColor={p[col]};labelBackgroundColor={p["bg"]};" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                     f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/><mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" '
                     f'as="targetPoint"/>{arr}</mxGeometry></mxCell>')
    return (f'<diagram id="{sc.key}" name="{html.escape(sc.title, quote=True)}"><mxGraphModel dx="{sc.w}" dy="{sc.h}" grid="1" '
            f'gridSize="10" page="1" pageWidth="{sc.w}" pageHeight="{sc.h}" background="{p["bg"]}"><root>'
            + ''.join(cells) + '</root></mxGraphModel></diagram>')


# where each scheme comes from (paths are relative to the repository root)
SOURCES = {
    'decode-layer': [
        'freetoken/python/freetoken/decode_overlap.py', 'freetoken/python/freetoken/models/qwen3_5_moe/moe.py',
        'freetoken/python/freetoken/layers/moe.py (forward_with_shared, _decode_routed)',
        'freetoken/python/freetoken/moe/offload_cache.py (ensure_experts, copy_missing, copy_planned)',
        'freetoken/python/freetoken/kernel/csrc/jit/decode_gather.cuh', 'freetoken/python/freetoken/moe/fused_nvfp4.py',
        'results/ladder/summary.json (nsys.decode_16k)'],
    'oproj': [
        'kernels/decode-oproj/src/qwen_oproj_gemv.cu', 'kernels/decode-oproj/README.md',
        'kernels/decode-oproj/scripts/probe_splitk.py', 'freetoken/python/freetoken/kernel/cuda_oproj_gemv.py',
        'results/ladder/ncu-kernels.json'],
    'moe-decode': [
        'kernels/decode-moe/src/qwen_moe_decode.cu', 'kernels/decode-moe/README.md',
        'freetoken/python/freetoken/kernel/cuda_moe_decode.py', 'freetoken/python/freetoken/moe/fused_nvfp4.py',
        'results/ladder/ncu-kernels.json'],
    'epilogue': [
        'freetoken/python/freetoken/moe/shared_epilogue.py', 'freetoken/python/freetoken/models/qwen3_5_moe/moe.py'],
    'prefill-routes': [
        'freetoken/python/freetoken/moe/offload_cache.py (select_prefill_route, _prefetch_split, selected_prefill_layer, '
        'prefetch_prefill_layer, borrow_tail_slots)',
        'freetoken/python/freetoken/layers/moe.py (_prefill_routed)', 'freetoken/python/freetoken/engine/layer_major.py',
        'freetoken/python/freetoken/moe/layer_major.py', 'freetoken/python/freetoken/kernel/batch_memcpy.py',
        'freetoken/python/freetoken/kernel/fast_index_copy.py', 'freetoken/python/freetoken/kernel/csrc/jit/decode_gather.cuh'],
    'prefill-timeline': [
        'freetoken/python/freetoken/moe/offload_cache.py (_init_prefill_overlap_buffers, _invalidate_prefill_buffer, '
        'begin_prefill, prefetch_prefill_layer, wait_prefill_layer, release_prefill_layer)',
        'freetoken/python/freetoken/layers/moe.py (_wait_prefill_overlap)', 'results/ladder/summary.json (L6, L7)'],
    'gdn-prefill': [
        'kernels/gdn/fla/qwen36_gdn_fla.cuh', 'kernels/gdn/qwen36_gdn.cu', 'freetoken/python/freetoken/kernel/cuda_gdn.py',
        'freetoken/python/freetoken/models/qwen3_5_moe/gdn.py', 'results/ladder/summary.json (L1, L2, nsys)'],
    'embed-budget': [
        'freetoken/python/freetoken/engine/host_embeddings.py', 'freetoken/python/freetoken/kernel/triton/host_rows.py',
        'freetoken/python/freetoken/layers/embedding.py', 'freetoken/python/freetoken/engine/cache_budget.py',
        'results/ladder/summary.json (memory, L3-L5)', 'results/ladder/notes.md'],
}


# ---------------------------------------------------------------------------------------------
# 1. decode step of one MoE layer by streams (T3 + S3 + D5)
# ---------------------------------------------------------------------------------------------
def scene_decode_layer():
    s = Scene('decode-layer', 'Шаг decode одного MoE-слоя по потокам: было (f1) и стало (main128)',
              'Один токен, один из 40 MoE-слоёв. Ширина блоков условная: схема показывает порядок и перекрытие, '
              'а не длительности. Весь шаг decode записан в один CUDA-граф, вилка и соединение потоков становятся '
              'рёбрами графа.', 1180, 810, SOURCES['decode-layer'])
    # --- was
    s.group(20, 98, 1140, 188, 'Было (f1): всё на одном потоке, подряд')
    s.text(34, 168, 'основной', 12, 600, 'fg')
    s.text(34, 184, 'поток', 12, 600, 'fg')
    row = [
        (150, 100, 'op', 'Роутер', 'gate bf16 → softmax, top-8'),
        (264, 160, 'op', 'Shared-эксперт', 'gate_up → SiLU·mul → down; линейка σ-гейта'),
        (438, 120, 'op', 'ensure_experts', 'LRU: попадание или слот вытеснения'),
        (572, 180, 'link', 'Подкачка промахов', 'ядро копирования: pinned RAM → слоты VRAM, ждёт PCIe'),
        (766, 220, 'op', 'Routed-эксперты', 'Triton: gate_up → SiLU·mul → down → Σ по top-8'),
        (1000, 150, 'old', 'Эпилог', '3 мелких bf16-ядра: σ(gate), shared·σ, routed + prod'),
    ]
    for x, w, kind, t, d in row:
        s.box(x, 128, w, 92, kind, t, d)
    for a, b in zip(row, row[1:]):
        s.arrow([(a[0] + a[1], 174), (b[0], 174)])
    s.arrow([(150, 240), (1150, 240)], width=1.2)
    s.text(1150, 258, 'время →', 11.5, 400, 'muted', 'end')
    s.line(264, 232, 752, 232, 'old', 2)
    s.text(508, 258, 'shared, затем подкачка: время слоя = сумма', 11.5, 400, 'old', 'middle')
    # --- now
    s.group(20, 300, 1140, 352, 'Стало (main128): подкачка идёт на побочном потоке, пока считается shared-эксперт')
    s.box(142, 326, 1008, 108, 'mem', '')
    s.box(142, 444, 1008, 84, 'mem', '')
    s.box(142, 538, 1008, 44, 'mem', '')
    s.text(34, 376, 'основной', 12, 600, 'fg')
    s.text(34, 392, 'поток', 12, 600, 'fg')
    s.text(34, 484, 'побочный', 12, 600, 'fg')
    s.text(34, 500, 'поток', 12, 600, 'fg')
    s.text(34, 564, 'канал PCIe', 12, 600, 'fg')
    s.box(152, 336, 100, 88, 'op', 'Роутер', 'gate bf16 → softmax, top-8')
    s.box(264, 336, 120, 88, 'op', 'ensure_experts', 'LRU: слоты для промахов, topk_ids → слоты')
    s.box(414, 336, 290, 88, 'lever', 'Shared-эксперт',
          'gate_up → SiLU·mul → down и линейка σ-гейта: считается, пока идёт подкачка')
    s.box(744, 336, 244, 88, 'lever', 'Routed: наше CUDA-ядро MoE',
          'gate_up → SiLU·mul → down (× вес роутера) → Σ по top-8')
    s.box(998, 336, 142, 88, 'lever', 'Слитый эпилог', 'σ(gate)·shared + routed одним ядром')
    s.arrow([(252, 380), (264, 380)])
    s.arrow([(384, 380), (414, 380)])
    s.arrow([(704, 380), (744, 380)])
    s.arrow([(988, 380), (998, 380)])
    s.box(414, 452, 290, 68, 'link', 'Подкачка промахов (TMA-ядро)',
          'cp.async.bulk: 8 CTA × 3 стадии × 16 КиБ, pinned RAM → слоты')
    s.box(414, 546, 290, 28, 'link', 'PCIe ≈ 42–49 ГБ/с: RAM → слоты VRAM', tsize=11.5)
    s.arrow([(324, 424), (324, 486), (414, 486)], 'sync', 'ready', 366, 480)
    s.arrow([(704, 486), (766, 486), (766, 424)], 'sync', 'done', 735, 480)
    s.arrow([(559, 520), (559, 546)], 'xfer')
    s.note(150, 592, 990, 'До соединения потоков слой ждёт max(shared, подкачка), а не их сумму. Если промахов нет, '
           'ядро подкачки выходит сразу: число промахов читается на устройстве, форма запуска фиксирована.', 11.5)
    s.text(1140, 322, 'пунктир с «ready» и «done» — события между потоками', 11, 400, 'sync', 'end')
    # --- kernels
    s.group(20, 664, 1140, 132, 'Что внутри routed: какие ядра наши, какие без изменений')
    chips = [(36, 'lever', 'gate_up GEMV (наше ядро)', 'N = 1024, K = 2048, NVFP4 → bf16 [8 × 1024]'),
             (312, 'op', 'SiLU·mul', 'ядро Triton, без изменений'),
             (588, 'lever', 'down GEMV (наше ядро)', 'N = 2048, K = 512, × вес роутера → [8 × 2048]'),
             (864, 'op', 'Σ по top-8', 'ядро Triton, без изменений')]
    for x, kind, t, d in chips:
        s.box(x, 696, 250, 56, kind, t, d)
    for i in range(3):
        s.arrow([(chips[i][0] + 250, 724), (chips[i + 1][0], 724)])
    s.text(36, 780, 'Nsight Systems, 16K (под профайлером): подкачка 1.08 → 0.64 мс из 7.07 → 5.78 мс на шаг decode; '
           'эпилог: −2 запуска на слой, −80 на шаг.', 11.5, 400, 'muted')
    return s


# ---------------------------------------------------------------------------------------------
# 2. o-projection (S3): cuBLASLt split-K = 3 versus libqwen_oproj_gemv
# ---------------------------------------------------------------------------------------------
def scene_oproj():
    s = Scene('oproj', 'o-проекция в decode: split-K = 3 у cuBLASLt и наше ядро в один запуск',
              'Y[M, 2048] = bf16((X[M, 4096] · W[2048, 4096]ᵀ) · sa · sb) в FP8 e4m3, M = 1…8. 40 таких проекций на шаг: '
              '30 out_proj слоёв GDN и 10 o_proj слоёв attention.', 1180, 960, SOURCES['oproj'])
    s.group(20, 90, 1140, 124, 'K = 4096 байт строки = 64 тайла по 64 байта; тайл t получает сплит z = t mod 3, по кругу')
    for t in range(64):
        s.tile(36 + t * 17, 120, 16, 26, f'z{t % 3}')
    for t, lab in ((0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5'), (62, '62'), (63, '63')):
        s.text(36 + t * 17 + 8, 162, lab, 10, 400, 'muted', 'middle')
    s.text(36 + 33 * 17, 162, '…', 11, 400, 'muted', 'middle')
    for x, kind, txt in ((36, 'z0', 'сплит 0: тайлы 0, 3, …, 63 (22 тайла)'), (360, 'z1', 'сплит 1: тайлы 1, 4, …, 61 (21)'),
                         (660, 'z2', 'сплит 2: тайлы 2, 5, …, 62 (21)')):
        s.tile(x, 178, 16, 16, kind)
        s.text(x + 24, 191, txt, 12, 400, 'fg')
    s.text(940, 191, 'не непрерывные куски', 12, 600, 'old')
    # --- was
    s.group(20, 228, 565, 566, 'Было: memset + GEMM xmma, split-K внутри ядра cuBLASLt')
    s.box(36, 262, 534, 44, 'old', '1. memset 384 Б', 'семафоры split-K')
    s.box(36, 316, 534, 62, 'old', '2. ядро xmma e4m3 → bf16, плитка 32×64×64',
          'сетка (1, 32, 3) = 96 CTA × 128 потоков: 32 блока по выходам (2048 / 32 = 64 выхода на блок) × 3 сплита K')
    for i, (x, kind) in enumerate(((36, 'z0'), (216, 'z1'), (396, 'z2'))):
        s.box(x, 400, 174, 112, kind, f'сплит {i}', f'тайлы {i}, {i + 3}, {i + 6}, …\nцепочка QMMA k32: 2 штуки на тайл, от +0 → p{i}')
        s.arrow([(x + 87, 378), (x + 87, 400)], width=1.2)
        s.arrow([(x + 87, 512), (x + 87, 540)], width=1.4)
    s.box(36, 540, 534, 74, 'old', '3. редукция внутри ядра, по очереди',
          'ожидание LDG.STRONG → семафор ATOMG.ADD → последовательные FADD; порядок (p0 + p2) + p1')
    s.box(36, 624, 534, 46, 'old', '4. эпилог', 'alpha = sa·sb; FFMA(acc, alpha, 0); BF16 с округлением к чётному')
    s.box(36, 684, 534, 96, 'mem', 'Что показал замер',
          'Nsight Compute (f1, базовые частоты, один запуск): 16.4 мкс, DRAM 58.1 %, SM 20.8 %, заполнение 11.2 %, '
          '74 регистра на поток. Микробенч на CUDA-графе вместе с memset: 17.4 мкс.')
    # --- now
    s.group(595, 228, 565, 566, 'Стало: libqwen_oproj_gemv, один запуск без memset и семафора')
    s.box(611, 262, 533, 62, 'lever', 'Сетка: полоса = 16 строк весов',
          '128 полос (2048 / 16). M = 1…3: 128 CTA × 96 потоков; M = 4…8: 64 CTA × 192 потока, по две полосы в CTA')
    s.group(611, 336, 533, 258, 'Один CTA: полоса из 16 строк W')
    sm = [(623, 'memhot', 'Веса W (VRAM)', 'на тайл 16 строк × 64 Б'),
          (754, 'mem', 'Кольца cp.async', '3 кольца, по одному на варп: 12 стадий (M ≤ 3) или 8 (M ≥ 4) × 1 КиБ'),
          (885, 'mem', 'Активация X', 'M × (K+64) Б в shared, переставлена под фрагменты MMA'),
          (1016, 'mem', 'Частичные p1, p2', 'в shared; их пишут варпы 1 и 2')]
    for x, kind, t, d in sm:
        s.box(x, 366, 113, 96, kind, t, d, tsize=12, dsize=10.5)
    s.arrow([(736, 414), (754, 414)], 'xfer')
    for i, (x, kind) in enumerate(((623, 'z0'), (797, 'z1'), (971, 'z2'))):
        s.box(x, 476, 160, 84, kind, f'варп {i} (z = {i})',
              f'тайлы {i}, {i + 3}, …: ldmatrix + 2 × QMMA k32; acc = строки n0…n0+7 и n0+8…n0+15', tsize=12, dsize=10.5)
    s.arrow([(703, 560), (703, 604)], width=1.4)
    s.arrow([(877, 560), (877, 604)], 'sync', 'p1', 890, 590, width=1.4)
    s.arrow([(1051, 560), (1051, 604)], 'sync', 'p2', 1064, 590, width=1.4)
    s.box(611, 604, 533, 50, 'lever', 'Варп 0 после __syncthreads',
          'out = BF16(FFMA((p0 + p2) + p1, sa·sb, 0)) → Y[m, n0…n0+15]', dsize=11)
    s.box(611, 668, 533, 112, 'memhot', 'Результат',
          'нет memset, семафора и рабочего буфера. Nsight Compute (main128, те же условия): 16.4 → 14.8 мкс, DRAM 58 → 66 %, '
          'SM 20.8 → 11.4 %. Микробенч на графе, M = 1: 17.4 → 12.1–12.5 мкс на проекцию (≈ 0.2 мс на шаг). '
          'Арифметика совпадает с cuBLASLt бит в бит.')
    # --- guarantees
    s.group(20, 808, 1140, 138, 'Как гарантируется бит-в-бит и что будет при несовпадении')
    gs = [(36, 'op', 'Выведено пробами', 'probe_splitk.py: слагаемые ±448·448 и 2⁻¹⁸ показывают, какие тайлы делят цепочку и в каком порядке складываются'),
          (260, 'op', 'Загрузка', 'библиотека из запечатанного memfd только при совпадении sha256 и ABI'),
          (484, 'op', 'Самопроверка', 'M = 1…8: коды, гауссовы, граничные, −0, масштабы; побитово с torch._scaled_mm'),
          (708, 'op', 'Проверка весов', 'реальные веса каждого слоя, для каждого включённого M'),
          (932, 'old', 'Несовпадение', 'этот M остаётся на cuBLASLt, событие пишется в лог')]
    for x, kind, t, d in gs:
        s.box(x, 838, 208, 94, kind, t, d, dsize=10.5)
    for a, b in zip(gs, gs[1:]):
        s.arrow([(a[0] + 208, 885), (b[0], 885)])
    return s


# ---------------------------------------------------------------------------------------------
# 3. MoE decode GEMV (S3)
# ---------------------------------------------------------------------------------------------
def scene_moe_decode():
    s = Scene('moe-decode', 'MoE decode: GEMV по NVFP4-экспертам, раскладка по блокам и потокам',
              'Те же операции и тот же порядок накопления, что у Triton-ядра; меняется только перемещение данных. '
              'Показан M = 1 (один токен, 8 экспертов), вариант ядра для M ≤ 2: 128 потоков, по 2 строки на команду.',
              1180, 1000, SOURCES['moe-decode'])
    s.group(20, 90, 1140, 154, 'Routed-часть MoE-блока на шаге decode')
    chips = [(36, 124, 'mem', 'Вход x', 'bf16 [2048]'),
             (190, 230, 'lever', 'gate_up GEMV', 'N = 1024, K = 2048; сетка (8, 16) × 128 потоков'),
             (444, 150, 'op', 'SiLU·mul', 'Triton, без изменений; [8 × 1024] → [8 × 512]'),
             (618, 250, 'lever', 'down GEMV', 'N = 2048, K = 512, × вес роутера; сетка (8, 32) × 128 потоков'),
             (892, 150, 'op', 'Σ по top-8', 'Triton, без изменений; [8 × 2048] → [2048]'),
             (1066, 78, 'mem', 'Выход', 'bf16')]
    for x, w, kind, t, d in chips:
        s.box(x, 124, w, 64, kind, t, d, dsize=10.5)
    for a, b in zip(chips, chips[1:]):
        s.arrow([(a[0] + a[1], 156), (b[0], 156)])
    s.note(36, 198, 1108, 'Сетка: blockIdx.x = маршрут (токен × 8 + k), blockIdx.y = полоса из 64 выходов; слот эксперта берётся из '
           'topk_ids (его проставил ensure_experts). Для M ≥ 3 полоса 128 выходов: сетки (M×8, 8) и (M×8, 16).', 11.5)
    # --- row split
    s.group(20, 258, 560, 372, 'Строка весов: куски по 16 байт и четыре потока команды')
    cols = ['z0', 'z1', 'z2', 'z3']
    for c in range(16):
        s.tile(36 + c * 31, 292, 30, 30, cols[c % 4], str(c), 10.5)
    s.text(36 + 16 * 31 + 2, 312, '… 63', 11, 400, 'muted')
    for it in range(4):
        x = 36 + it * 4 * 31
        s.line(x + 2, 330, x + 4 * 31 - 3, 330, 'flow', 1.4)
        s.text(x + 2 * 31, 346, f'it = {it}', 10.5, 400, 'muted', 'middle')
    s.text(36 + 16 * 31 + 2, 346, 'it = 15', 10.5, 400, 'muted')
    for i, (x, k) in enumerate(((36, 'z0'), (162, 'z1'), (288, 'z2'), (414, 'z3'))):
        s.tile(x, 360, 14, 14, k)
        s.text(x + 20, 372, f'r{i}: c = 4·it + {i}', 11, 400, 'fg')
    s.box(36, 388, 252, 108, 'mem', 'Кусок c = 16 Б весов + 2 Б масштабов',
          '32 кода e2m1 (4 слова по 8) = K от 32c до 32c + 31; 2 масштаба e4m3 на блоки по 16 значений', dsize=10.5)
    s.box(298, 388, 270, 108, 'mem', 'Кольцо предвыборки в регистрах',
          'PF = 2 куска в полёте на поток: ld.global.nc.v4 (16 Б) и ld u16 для масштабов, без выделения в L1; '
          'веса запрашиваются раньше, чем активация попадает в shared', dsize=10.5)
    s.box(36, 508, 532, 110, 'lever', 'CTA: 128 потоков = 32 команды × 4 потока',
          'команда t считает строки n0 + t и n0 + t + 32 (2 строки на команду); поток r команды берёт куски c = 4·it + r. '
          'Активация bf16 → fp32 кладётся в shared один раз на CTA (K × 4 Б = 8 КиБ при K = 2048).', dsize=11)
    # --- per-thread
    s.group(600, 258, 560, 372, 'Что делает поток r с куском c (порядок как у Triton)')
    s.box(616, 292, 528, 56, 'op', '1. Слово w = 8 кодов e2m1',
          'cvt e2m1 → f16 → f32: точная таблица, код 8 = −0.0; активация a из shared (fp32)', dsize=11)
    s.box(616, 358, 528, 56, 'lever', '2. Цепочка FFMA от +0',
          'acc = 0; acc = fma(a[8w + j], код_j, acc), j = 0…7; произведения точны', dsize=11)
    s.box(616, 424, 528, 56, 'lever', '3. Частичная сумма',
          'P[4r + q] = fma(acc, масштаб блока, P[4r + q]), q = 0…3: 16 частичных сумм P на строку', dsize=11)
    for y in (348, 414):
        s.arrow([(880, y), (880, y + 10)], width=1.3)
    s.box(616, 494, 528, 124, 'mem', 'Что было узким местом и что стало',
          'Triton-ядро в f1 (Nsight Compute, базовые частоты, один запуск): сетка (8, 64, 1) × 128 потоков, 23.0 мкс, '
          'DRAM 47.6 %, SM 64.0 %, заполнение 53.5 %, 40 регистров: SM загружены сильнее памяти. '
          'Наше ядро (main128): 17.8 мкс, DRAM 64.5 %, SM 18.7 %, заполнение 15.1 %: память загружена сильнее SM.\n'
          'Микробенч на графе, M = 1: gate_up 18.5 → 12.5–12.9 мкс, down 9.4 → 7.4–7.8 мкс.', dsize=10.5)
    # --- tree
    s.group(20, 644, 1140, 338, 'Редукция 16 частичных сумм: то же дерево, что у Triton, через shfl.xor')
    xs = [36, 196, 356, 516]
    for i, x in enumerate(xs):
        k = cols[i]
        s.box(x, 678, 150, 44, k, f'поток r{i}', f'P[{4 * i}] … P[{4 * i + 3}]', dsize=10.5)
        s.box(x, 738, 150, 50, k, f's{2 * i} = P[{4 * i}] + P[{4 * i + 1}]', f's{2 * i + 1} = P[{4 * i + 2}] + P[{4 * i + 3}]',
              tsize=11, dsize=11)
        s.arrow([(x + 75, 722), (x + 75, 738)], width=1.3)
    s.box(36, 818, 150, 50, 'z0', 't0 = s0 + s4', 't1 = s1 + s5', tsize=11, dsize=11)
    s.box(196, 818, 150, 50, 'z1', 't0 = s2 + s6', 't1 = s3 + s7', tsize=11, dsize=11)
    s.arrow([(111, 788), (111, 818)], 'sync', 'shfl.xor 2', 74, 810, width=1.4)
    s.arrow([(431, 788), (431, 808), (140, 808), (140, 818)], 'sync', '', width=1.3)
    s.arrow([(591, 788), (591, 800), (260, 800), (260, 818)], 'sync', '', width=1.3)
    s.arrow([(271, 788), (271, 818)], 'sync', '', width=1.3)
    s.box(36, 898, 310, 56, 'z0', 'u0 = (s0+s4) + (s2+s6)', 'u1 = (s1+s5) + (s3+s7)', tsize=11, dsize=11)
    s.arrow([(111, 868), (111, 898)], 'sync', 'shfl.xor 1', 74, 890, width=1.4)
    s.arrow([(271, 868), (271, 878), (200, 878), (200, 898)], 'sync', '', width=1.3)
    s.box(376, 898, 304, 56, 'lever', 'v = u0 + u1, поток r0',
          'out = bf16(v × global[слот, n]) (в down ещё × вес роутера) → c[токен, k, n]', tsize=11, dsize=10.5)
    s.arrow([(346, 926), (376, 926)])
    s.note(716, 672, 428, 'shfl.xor 2 складывает потоки r0 с r2 и r1 с r3, shfl.xor 1 — r0 с r1; итог собирает поток r0. '
           'Каждое сложение, умножение и fma прописаны в PTX с явным округлением (add.rn, mul.rn, fma.rn): '
           'ptxas не может ни слить их, ни переставить. Произведение e2m1 × bf16 точно в fp32, поэтому цепочка FFMA равна '
           'FMUL + FADD бит в бит.', 11.5, 'muted')
    s.box(716, 780, 428, 84, 'memhot', 'Проверка',
          '4284 сравнения GEMM, 0 расхождений; самопроверка при загрузке (каждый код e2m1 и масштаб e4m3, M = 1…8, top_k 8) '
          'и проверка на реальных весах каждого слоя.', dsize=10.5)
    s.box(716, 874, 428, 80, 'old', 'При несовпадении', 'слой остаётся на Triton-ядре, событие пишется в лог; точность сохраняется, выигрыш пропадает.', dsize=10.5)
    return s


# ---------------------------------------------------------------------------------------------
# 4. fused epilogue (D5)
# ---------------------------------------------------------------------------------------------
def scene_epilogue():
    s = Scene('epilogue', 'Слитый эпилог MoE-блока: три мелких ядра → одно',
              'out = routed + shared · σ(gate). Те же операции и те же округления, что у torch; fp-contraction выключен.',
              1180, 500, SOURCES['epilogue'])
    s.group(20, 78, 1140, 134, 'Было: три bf16-ядра PyTorch, каждое читает и пишет [M, 2048] в память')
    s.box(36, 112, 156, 80, 'mem', 'gate', 'линейка σ-гейта, bf16 [M, 1]')
    s.box(216, 112, 270, 80, 'old', 'ядро 1: torch.sigmoid', 'sig = bf16(σ(gate))')
    s.box(510, 112, 300, 80, 'old', 'ядро 2: shared · sig', 'prod = bf16(shared × sig)')
    s.box(834, 112, 310, 80, 'old', 'ядро 3: routed + prod', 'out = bf16(routed + prod)')
    for a, b in ((192, 216), (486, 510), (810, 834)):
        s.arrow([(a, 152), (b, 152)])
    s.group(20, 226, 1140, 262, 'Стало: одно ядро Triton, 4 варпа; сетка (M, ⌈2048 / 1024⌉), при M = 1 это (1, 2)')
    s.box(36, 260, 200, 76, 'mem', 'Читает', 'gate (скаляр на строку), shared[h], routed[h]; всё bf16', dsize=10.5)
    s.box(256, 260, 270, 76, 'lever', 'sig = bf16(1 / (1 + exp(−gate)))', 'IEEE-деление div.rn, expf из libdevice', tsize=11.5)
    s.box(546, 260, 270, 76, 'lever', 'prod = bf16(shared × sig)', 'одно умножение fp32, затем округление в bf16', tsize=11.5)
    s.box(836, 260, 308, 76, 'lever', 'out = bf16(routed + prod)', 'одно сложение fp32, затем округление; запись [M, 2048]', tsize=11.5)
    for a, b in ((236, 256), (526, 546), (816, 836)):
        s.arrow([(a, 298), (b, 298)])
    s.box(36, 352, 560, 120, 'old', 'Почему enable_fp_fusion = False',
          'С настройкой Triton по умолчанию умножение и сложение сливаются в один fma.rn.bf16: одно округление вместо двух, '
          'и результат расходится с torch. Здесь два округления остаются раздельными, как в трёх ядрах torch.', dsize=11)
    s.box(616, 352, 528, 120, 'memhot', 'Самопроверка до первого вызова',
          'σ сверяется побитово с torch.sigmoid на всех 65 536 значениях bf16; затем случайные M = 1, 3, 8 и H = 2048, 1000 '
          'на трёх масштабах. Несовпадение, или вызов внутри захвата графа до проверки, оставляет путь torch.', dsize=11)
    return s


# ---------------------------------------------------------------------------------------------
# 5. prefill transport: three routes into the double buffer (T1 + C1 + layer-major)
# ---------------------------------------------------------------------------------------------
def scene_prefill_routes():
    s = Scene('prefill-routes', 'Prefill: как эксперты слоя попадают в двойной буфер (три маршрута)',
              'Слой экспертов NVFP4 весит ≈ 456 МБ (256 экспертов). Двойной буфер занимает первые 2 × 256 слотов кэша экспертов; '
              'GEMM слоя L читает буфер L mod 2. Маршрут выбирает число некэшированных токенов запроса; цвет стрелок совпадает '
              'с цветом карточек ниже.', 1180, 920, SOURCES['prefill-routes'])
    s.group(20, 96, 300, 316, 'RAM хоста (pinned)')
    s.box(36, 134, 268, 110, 'mem', 'Банки экспертов NVFP4', '40 слоёв × 256 экспертов; слой ≈ 456 МБ, всего 18.25 ГБ')
    s.box(36, 262, 268, 120, 'mem', 'Снимок карты слотов', '[40, 256] int32 в pinned-памяти: один на чанк, нужен маршруту 2')
    s.box(338, 134, 78, 248, 'link', 'PCIe 5.0 x16', '≈ 41–49 ГБ/с', dsize=10.5)
    s.group(436, 96, 724, 316, 'VRAM: кэш слотов экспертов (4603 слота)')
    s.box(452, 134, 160, 80, 'memhot', 'Буфер 0', 'слоты 0…255: слои 0, 2, 4, …')
    s.box(452, 226, 160, 80, 'memhot', 'Буфер 1', 'слоты 256…511: слои 1, 3, 5, …')
    s.box(648, 134, 250, 172, 'mem', '«Горячие» эксперты decode',
          'слоты 512…4602: то, что осталось в кэше после decode; источник копий внутри GPU')
    s.box(914, 134, 230, 172, 'mem', 'Хвост слотов (layer-major)',
          'hidden + residual окна чанков: 8 КиБ на токен; слоты на это время размечаются как пустые')
    s.box(452, 322, 692, 76, 'op', 'GEMM экспертов слоя L',
          'читает буфер L mod 2; номер строки буфера = номер эксперта, поэтому ids роутера не переписываются')
    s.arrow([(304, 188), (338, 188)], 'xfer')
    for base in (150, 242):
        for dy, kind in ((0, 'a0'), (16, 'a1'), (32, 'a2')):
            s.arrow([(416, base + dy), (452, base + dy)], kind, width=2.2)
    for y in (150, 242):
        s.arrow([(648, y), (612, y)], 'a0', width=2.2)
        s.arrow([(648, y + 16), (612, y + 16)], 'a1', width=2.2)
    s.text(630, 144, 'D2D', 10.5, 600, 'muted', 'middle')
    s.text(630, 236, 'D2D', 10.5, 600, 'muted', 'middle')
    cards = [
        ('z0', 428, '1. Короткий кусок, меньше 160 токенов: только выбранные роутером эксперты (раньше: слой целиком)',
         [('Роутер слоя L', 'выбранные эксперты известны только после роутера, поэтому копия начинается после него'),
          ('Два небольших ядра', 'помечают выбранных и делят на попадания (слот ≥ 512) и промахи; списки остаются на устройстве'),
          ('Попадания и промахи', 'попадания: fast_index_copy на потоке вычислений; промахи: TMA-gather (cp.async.bulk) на потоке копирования'),
          ('GEMM после события ready', 'читает только выбранные строки буфера; байты остальных строк не читаются')]),
        ('z1', 586, '2. Чанк 160…2048 токенов: попадания копируются внутри GPU, промахи по PCIe (раньше: весь слой по PCIe)',
         [('Снимок карты слотов', 'в начале чанка, в pinned-память: одна синхронизация на чанк'),
          ('Старт слоя L: заказ слоя L+1', 'по снимку хост делит экспертов на попадания (слот ≥ 512) и промахи'),
          ('Попадания и промахи', 'fast_index_copy (64 блока на банк) на потоке вычислений; отрезки подряд идущих экспертов — один cudaMemcpyBatchAsync'),
          ('События release и ready', 'release после GEMM освобождает буфер; ready разрешает GEMM следующего слоя')]),
        ('z2', 744, '3. Длинный запрос, от 2049 токенов: послойный prefill (в исходном FreeToken слой копировался заново для каждого чанка)',
         [('Чанк 0 слоя L', 'заказывает копию слоя L+1 целиком в свободный буфер'),
          ('cudaMemcpyAsync по банкам', 'copy engine, один раз на запрос: на 120K 18 ГБ вместо 1076 ГБ'),
          ('Все чанки слоя L', 'читают один буфер; hidden и residual окна лежат в хвосте слотов'),
          ('release после последнего чанка', 'копия слоя L+2 ждёт это событие; попаданий нет, горячие эксперты не используются')]),
    ]
    for kind, y, title, steps in cards:
        s.group(20, y, 1140, 148, title, kind)
        xs = [36, 318, 600, 882]
        for x, (t, d) in zip(xs, steps):
            s.box(x, y + 36, 262, 100, kind, t, d, tsize=12, dsize=10.5)
        for a, b in zip(xs, xs[1:]):
            s.arrow([(a + 262, y + 86), (b, y + 86)], 'flow')
    return s


# ---------------------------------------------------------------------------------------------
# 6. copy overlapped with the GEMM (C0): before / after
# ---------------------------------------------------------------------------------------------
def scene_prefill_timeline():
    s = Scene('prefill-timeline', 'Копия слоя L+1 вместе с GEMM слоя L: что изменили приоритет потока и инвалидация раз за эпоху',
              'Двойной буфер: слой L+1 едет в буфер, который освободил слой L−1, пока считается слой L. Масштаб условный, показана '
              'механика. На коротком запросе копия слоя дольше счёта, поэтому задержка старта копии видна в TTFT.',
              1180, 800, SOURCES['prefill-timeline'])

    def panel(y0, title):
        s.group(20, y0, 1140, 244, title)
        s.text(34, y0 + 78, 'поток', 12, 600, 'fg')
        s.text(34, y0 + 94, 'вычислений', 12, 600, 'fg')
        s.text(34, y0 + 176, 'поток', 12, 600, 'fg')
        s.text(34, y0 + 192, 'копирования', 12, 600, 'fg')
        yc, yk = y0 + 52, y0 + 150
        s.box(140, yc - 6, 1010, 66, 'mem', '')
        s.box(140, yk - 6, 1010, 66, 'mem', '')
        return yc, yk

    # was
    yc, yk = panel(96, 'Было: перед каждой копией на потоке копирования стоят 6 мелких ядер инвалидации')
    s.box(150, yc, 90, 54, 'op', 'GEMM L−1', '', tsize=12)
    s.box(240, yc, 90, 54, 'op', 'GEMM L', '', tsize=12)
    s.ghost(330, yc, 190, 54, 'ждёт ready: копия L+1 ещё не закончена', 'old')
    s.box(520, yc, 90, 54, 'op', 'GEMM L+1', '', tsize=12)
    s.ghost(610, yc, 190, 54, 'ждёт ready', 'old')
    s.box(800, yc, 90, 54, 'op', 'GEMM L+2', '', tsize=12)
    s.ghost(240, yk, 90, 54, '6 ядер ждут SM', 'old', 10.5)
    s.box(330, yk, 190, 54, 'link', 'DMA слоя L+1', '≈ 456 МБ', dsize=10.5)
    s.ghost(520, yk, 90, 54, '6 ядер ждут SM', 'old', 10.5)
    s.box(610, yk, 190, 54, 'link', 'DMA слоя L+2', '≈ 456 МБ', dsize=10.5)
    s.ghost(800, yk, 90, 54, '6 ядер ждут SM', 'old', 10.5)
    s.box(890, yk, 190, 54, 'link', 'DMA слоя L+3', '', tsize=12)
    s.arrow([(240, yc + 54), (240, yk)], 'sync', 'release', 266, yc + 80, width=1.3)
    s.arrow([(520, yk), (520, yc + 54)], 'sync', 'ready', 546, yc + 80, width=1.3)
    s.text(1146, 96 + 232, 'период слоя = счёт + копия', 12, 600, 'old', 'end')
    # now
    yc, yk = panel(354, 'Стало: приоритет потока копирования, инвалидация буферов раз за эпоху на потоке вычислений')
    s.text(150, 354 + 40, 'один раз на эпоху, до первой копии: инвалидация обоих буферов на потоке вычислений', 11, 400, 'lever')
    s.box(150, yc, 90, 54, 'op', 'GEMM L−1', '', tsize=12)
    s.box(240, yc, 90, 54, 'op', 'GEMM L', '', tsize=12)
    s.ghost(330, yc, 100, 54, 'ждёт ready', 'old')
    s.box(430, yc, 90, 54, 'op', 'GEMM L+1', '', tsize=12)
    s.ghost(520, yc, 100, 54, 'ждёт ready', 'old')
    s.box(620, yc, 90, 54, 'op', 'GEMM L+2', '', tsize=12)
    s.box(240, yk, 190, 54, 'link', 'DMA слоя L+1', '≈ 456 МБ', dsize=10.5)
    s.box(430, yk, 190, 54, 'link', 'DMA слоя L+2', '', tsize=12)
    s.box(620, yk, 190, 54, 'link', 'DMA слоя L+3', '', tsize=12)
    s.arrow([(240, yc + 54), (240, yk)], 'sync', 'release', 266, yc + 80, width=1.3)
    s.arrow([(430, yk), (430, yc + 54)], 'sync', 'ready', 456, yc + 80, width=1.3)
    s.text(1146, 354 + 232, 'период слоя = max(счёт, копия)', 12, 600, 'good', 'end')
    s.text(840, yk + 24, 'копии идут подряд,', 11.5, 400, 'good')
    s.text(840, yk + 40, 'PCIe не простаивает', 11.5, 400, 'good')
    # mechanics
    s.group(20, 612, 1140, 176, 'Что именно изменено и почему это работает')
    s.box(36, 644, 270, 130, 'mem', 'Двойной буфер',
          'буфер 0 — слои 0, 2, 4, …; буфер 1 — слои 1, 3, 5, … Копия слоя L+1 пишет в буфер, который прочитал слой L−1: '
          'ей ждать нужно только release после GEMM слоя L−1.', dsize=10.5)
    s.box(318, 644, 270, 130, 'old', 'Причина задержки',
          'перед DMA на потоке копирования шли 6 мелких ядер инвалидации карты слотов. Им нужны SM, а GEMM слоя L занимает GPU; '
          'DMA стоит за ними в том же потоке.', dsize=10.5)
    s.box(600, 644, 270, 130, 'lever', 'Что сделали',
          'поток копирования получил высший приоритет (−1); оба буфера инвалидируются один раз на эпоху на потоке вычислений, '
          'в потоке копирования остаётся чистый DMA.', dsize=10.5)
    s.box(882, 644, 262, 130, 'memhot', 'Эффект в лесенке',
          'TTFT 64 токена 0.270 → 0.252 с, 2K 0.419 → 0.383 с. На 16K и 120K копия и так спрятана за счётом слоя, TTFT не меняется.',
          dsize=10.5)
    return s


# ---------------------------------------------------------------------------------------------
# 7. GDN prefill, FLA wu (T2)
# ---------------------------------------------------------------------------------------------
def scene_gdn_prefill():
    s = Scene('gdn-prefill', 'GDN в prefill: четыре ядра Triton → одно CUDA-ядро по чанкам',
              'Слой Gated DeltaNet: 32 головы значений, 16 голов ключей, размер головы 128, чанк 64 токена. Ядра conv + L2, '
              'gate + cumsum и gated norm + FP8 остались как в f1, kkt_solve остался на Triton.', 1180, 972, SOURCES['gdn-prefill'])
    s.group(20, 90, 1140, 214, 'Было: цепочка FLA на Triton, промежуточные тензоры проходят через память')
    ks = [(36, 'op', 'kkt_solve'), (262, 'old', 'recompute_w_u'), (488, 'old', 'chunk_h'), (714, 'old', 'chunk_o')]
    under = ['пишет A_wy [T, 32, 64] bf16', 'читает A_wy, k, v; пишет w и u [T, 32, 128] bf16',
             'читает k, w, u; пишет h (bf16 на каждый чанк) и v_new', 'читает q, k, v_new, h; пишет o [T, 32, 128]']
    for (x, kind, t), u in zip(ks, under):
        s.box(x, 126, 190, 56, kind, t, '', tsize=13)
        s.box(x, 206, 190, 82, 'mem', '', u, dsize=11)
        s.arrow([(x + 95, 182), (x + 95, 206)], 'xfer')
    for a, b in zip(ks, ks[1:]):
        s.arrow([(a[0] + 190, 154), (b[0], 154)])
    s.box(940, 126, 204, 162, 'mem', 'Итог', '4 запуска; w, u, h и v_new пишутся в память и читаются обратно', dsize=11)
    s.arrow([(904, 154), (940, 154)])
    s.group(20, 318, 1140, 168, 'Стало (вариант wu): kkt_solve и одно слитое CUDA-ядро')
    s.box(36, 354, 190, 56, 'op', 'kkt_solve', 'Triton, без изменений', tsize=13)
    s.box(262, 354, 628, 56, 'lever', 'chunk_wu_ho_fused: recompute_w_u + chunk_h + chunk_o',
          'сетка (2 блока V по 64, 32 головы, N последовательностей), 128 потоков (4 варпа), shared 99 328 Б', tsize=13)
    s.arrow([(226, 382), (262, 382)])
    s.box(262, 420, 628, 54, 'mem', '', 'читает q, k, v, A_wy, β, g; пишет только o и итоговое состояние H в пул состояний; '
          'w, u, h и v_new в глобальную память не попадают', dsize=11)
    s.box(940, 354, 204, 120, 'memhot', 'Итог', '2 запуска вместо 4; результат побитово равен цепочке Triton (teacher 125/125 на ступени)', dsize=11)
    s.arrow([(890, 382), (940, 382)])
    s.group(20, 500, 1140, 458, 'Внутри одного CTA: он идёт по чанкам своей последовательности подряд, по 64 токена')
    s.box(36, 536, 340, 112, 'memhot', 'Регистры: состояние H (fp32)',
          'варп w держит строки V [16w, 16w + 16) × K = 128: 64 регистра на поток, H^T лежит как аккумуляторы MMA; '
          'читается из пула один раз и пишется в конце', dsize=11)
    s.box(36, 660, 340, 142, 'mem', 'Shared, ≈ 97 КиБ',
          'w [64 × 128] (он же буфер для o); q [64 × 128]; k ×2; A_wy ×2; v [64 × 64]; Ag [64 × 64]; g и β ×2. '
          'k, A_wy, g, β — двойная буферизация cp.async; q и v — одинарная, перезаливаются сразу после использования.', dsize=10.5)
    s.box(36, 814, 340, 128, 'mem', 'Условия применения',
          'головы 16/32/128, T > 32, FP32-пул состояний; иначе остаётся Triton. Пустая последовательность в батче отправляет '
          'батч на старый путь.', dsize=10.5)
    ph = [('Фаза 0', 'w = A_wy · (k · β · e^g): цепочка MMA по 64 токенам, округление в bf16, w остаётся в shared'),
          ('Фаза 1', 'Ag = маска и затухание от q · kᵀ; v_dot = Hb · wᵀ и qh = Hb · qᵀ по K = 128, где Hb = bf16(H)'),
          ('Фаза 2', 'u = A_wy · (v · β); vn = u − v_dot; v_new = bf16(vn); vs = bf16(vn · e^(g_last − g))'),
          ('Фаза 3', 'o = bf16(fma(scale, qh · e^g, scale · (Ag · v_new))) → shared → запись; D = kᵀ · vs; H = fma(H, e^g_last, D)')]
    y = 536
    for i, (t, d) in enumerate(ph):
        s.box(400, y, 744, 58, 'lever', t, d, dsize=11, align='left')
        if i:
            s.arrow([(772, y - 8), (772, y)], width=1.3)
        y += 66
    s.arrow([(772, y - 8), (772, y)], width=1.3)
    s.box(400, y, 744, 48, 'op', 'Следующий чанк', 'q, v и k / A_wy / g / β уже загружены cp.async; H остался в регистрах', dsize=11, align='left')
    y += 62
    s.box(400, y, 744, 90, 'memhot', 'Что это дало',
          'GDN в prefill 16K: 67 → 51 мс (Nsight Systems); TTFT 16K 1.343 → 1.316 с (−2 %). Порядок операций в чанке '
          'не меняется: каждый выход видит те же наборы K и ту же цепочку, что в Triton.', dsize=11, align='left')
    return s


# ---------------------------------------------------------------------------------------------
# 8. embeddings in pinned RAM and the VRAM budget
# ---------------------------------------------------------------------------------------------
def scene_embed_budget():
    s = Scene('embed-budget', 'Эмбеддинги в pinned RAM и бюджет VRAM: откуда берутся лишние слоты экспертов',
              'Таблица эмбеддингов 248 320 × 2048 bf16 весит 0.95 ГиБ. Пока она в VRAM, эти байты вычитаются из бюджета слотов.',
              1180, 612, SOURCES['embed-budget'])
    s.group(20, 82, 560, 330, 'Строка эмбеддинга читается прямо из RAM')
    s.box(36, 118, 160, 58, 'mem', 'ids токенов', 'int в VRAM', dsize=11)
    s.box(228, 118, 330, 58, 'lever', 'Triton host_gather_rows', 'одна программа на токен, 4 варпа', dsize=11)
    s.arrow([(196, 147), (228, 147)])
    s.box(36, 214, 260, 84, 'mem', 'Таблица эмбеддингов в pinned RAM',
          '248 320 × 2048 bf16 = 0.95 ГиБ, отображена в адресное пространство GPU', dsize=10.5)
    s.box(328, 214, 230, 84, 'memhot', 'Строки в VRAM', '4096 Б на токен, байт в байт: целыми 8-байтными словами, без float', dsize=10.5)
    s.arrow([(262, 214), (262, 176)], 'xfer')
    s.text(272, 200, 'по PCIe: 4096 Б на строку', 10.5, 400, 'warn')
    s.arrow([(443, 176), (443, 214)])
    s.note(36, 316, 520, 'Идентификатор вне таблицы даёт нули. Если lm_head читает ту же таблицу (tied), перенос в RAM отклоняется: '
           'такая таблица нужна на GPU на каждом шаге.', 11.5)
    s.group(600, 82, 560, 330, 'Бюджет VRAM → число слотов экспертов')
    s.box(616, 118, 528, 84, 'op', 'пул = memory_ratio × свободная VRAM до модели − веса − фиксированный кэш',
          'остаток (1 − memory_ratio) — запас под CUDA-графы и активации; эксперты берут пул первыми (после резерва KV), KV получает остаток',
          tsize=12, dsize=10.5)
    scale = 512 / 4611
    xa, xb, xc = 616, 616 + 3164 * scale, 616 + 3729 * scale
    s.text(1144, 224, 'итого 4611 слотов', 11.5, 600, 'fg', 'end')
    s.tile(xa, 232, 3164 * scale, 40, 'mem', 'f1: 3164 слота', 12)
    s.tile(xb, 232, 565 * scale, 40, 'lever', '', 12)
    s.tile(xc, 232, 882 * scale, 40, 'memhot', '+882', 12)
    s.line(xb + 2, 272, xb + 2, 300, 'flow')
    s.line(xc + 2, 272, xc + 2, 292, 'flow')
    s.note(616, 286, 342, '+565 слотов: таблица эмбеддингов ушла в RAM, освободилось 0.95 ГиБ', 11.5, 'lever', 600, 'end')
    s.note(xc + 8, 280, 112, 'memory_ratio 0.75 → 0.85', 11.5, 'good', 600)
    s.note(616, 346, 528, 'В main128 слотов 4603: самопроверки ядер при финализации весов оставляют ≈ 14 МиБ в кэше аллокатора, '
           'и бюджет засчитывает их как веса. Плотные веса FP8 и attention ≈ 1.57 ГиБ, KV-кэш на 128K ≈ 2.5 ГиБ.', 11)
    s.group(20, 426, 1140, 160, 'Что это дало (лесенка)')
    s.box(36, 462, 270, 108, 'memhot', 'Промахи кэша на токен, 16K', '27.7 → 19.9 → 12.3 при слотах 3164 → 3729 → 4611', dsize=11)
    s.box(318, 462, 270, 108, 'lever', 'Эмбеддинги в RAM', 'декод 16K +5.2 ток/с (148.6 → 153.8); TTFT в пределах шума', dsize=11)
    s.box(600, 462, 270, 108, 'lever', 'memory_ratio 0.85', 'декод 16K +5.7 ток/с (153.8 → 159.5); TTFT 64 токена 0.314 → 0.259 с', dsize=11)
    s.box(882, 462, 262, 108, 'mem', 'Windows не подкачивает', 'за 2-часовой soak shared-память Windows выросла на 30 МиБ', dsize=11)
    return s


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
SCENES = [scene_decode_layer, scene_oproj, scene_moe_decode, scene_epilogue, scene_prefill_routes,
          scene_prefill_timeline, scene_gdn_prefill, scene_embed_budget]


def write(path, text):
    path.write_text(text, encoding='utf-8', newline='')  # text already carries CRLF


def main():
    scenes = [f() for f in SCENES]
    if '--sources' in sys.argv:
        for sc in scenes:
            print(sc.key)
            for src in sc.sources:
                print('   ', src)
        return 0
    img = ROOT / 'docs' / 'img'
    img.mkdir(parents=True, exist_ok=True)
    for sc in scenes:
        write(img / f'kernel-{sc.key}.svg', render_svg(sc, 'light'))
        write(img / f'kernel-{sc.key}-dark.svg', render_svg(sc, 'dark'))
    pages = ''.join(drawio_page(sc) for sc in scenes)
    write(ROOT / 'docs' / 'kernels-lowlevel.drawio',
          '<mxfile host="tools/kernels-diagram.py">' + pages + '</mxfile>' + NL)
    for w in WARNINGS:
        print('WARNING', w)
    print(f'{len(scenes)} schemes written to docs/img/kernel-*.svg and docs/kernels-lowlevel.drawio')
    return 1 if WARNINGS else 0


if __name__ == '__main__':
    sys.exit(main())
