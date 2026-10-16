#!/usr/bin/env python3
"""Block diagram of Qwen3.6-35B-A3B NVFP4 on a 16 GB card, as served by main128.

One description, three outputs: docs/img/model-blocks.svg, docs/img/model-blocks-dark.svg (README) and
docs/model-blocks.drawio (open and edit in draw.io). Stdlib only, deterministic.
Usage: python tools/model-diagram.py
"""
import html
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
W, H = 1180, 1010

# id: (x, y, w, h, kind, title, detail)
BOXES = {
    'tok':   (40, 60, 330, 54, 'io', 'Токены запроса', 'chat-шаблон; prefix-кэш отрезает уже посчитанное'),
    'emb':   (40, 140, 330, 64, 'lever', 'Эмбеддинги  248 320 × 2048, BF16', 'таблица в pinned RAM, строки копируются на GPU побайтно'),
    'layer': (20, 240, 370, 560, 'group', '× 40 слоёв декодера', ''),
    'norm1': (40, 290, 330, 40, 'op', 'RMSNorm', ''),
    'gdn':   (40, 350, 158, 84, 'lever', '30 × GDN', 'линейное внимание; CUDA-ядра FLA (T2)'),
    'attn':  (212, 350, 158, 84, 'op', '10 × attention', 'полное внимание, KV BF16'),
    'norm2': (40, 454, 330, 40, 'lever', 'o-proj FP8 GEMV (S3) + residual → RMSNorm', ''),
    'router': (40, 514, 330, 50, 'op', 'Роутер: top-8 из 256 экспертов', 'bf16'),
    'lru':   (40, 584, 330, 50, 'op', 'Карта слотов (LRU)', 'попадание → слот VRAM, промах → gather из RAM'),
    'routed': (40, 654, 158, 70, 'lever', 'Routed GEMM NVFP4', 'decode: CUDA MoE (S3)'),
    'shared': (212, 654, 158, 70, 'lever', 'Shared-эксперт', 'параллельно с gather (T3)'),
    'epi':   (40, 744, 330, 40, 'lever', 'Эпилог: routed + σ(gate)·shared — одно ядро (D5)', ''),
    'head':  (40, 830, 330, 54, 'op', 'Финальная норма → lm_head NVFP4', ''),
    'out':   (40, 910, 330, 54, 'io', 'Сэмплер → токен ответа', 'декод в CUDA-графе'),
    # memory side
    'vram':  (460, 60, 330, 360, 'group', 'VRAM 16 ГБ  (RTX 5070 Ti)', ''),
    'win':   (480, 106, 290, 40, 'mem', 'Windows — ровно 1 ГиБ', ''),
    'w':     (480, 156, 290, 40, 'mem', 'Плотные веса FP8 и attention  ≈ 1,8 ГиБ', ''),
    'kv':    (480, 206, 290, 40, 'mem', 'KV-кэш BF16 + пул GDN  ≈ 3 ГиБ', ''),
    'slots': (480, 256, 290, 64, 'memhot', 'Слоты экспертов: 4603', '≈ 7,6 ГиБ, «горячие» эксперты всех слоёв (f1: 3164)'),
    'buf':   (480, 330, 290, 64, 'mem', 'Двойной буфер слоя для prefill', 'сюда приходит банк следующего слоя'),
    'ram':   (460, 560, 330, 224, 'group', 'RAM хоста', ''),
    'banks': (480, 606, 290, 64, 'mem', 'Банки экспертов NVFP4  18,25 ГБ', 'все 256 × 40, закреплённая память'),
    'embh':  (480, 690, 290, 40, 'mem', 'Таблица эмбеддингов  0,95 ГиБ', ''),
    'pcie':  (520, 452, 210, 64, 'link', 'PCIe  ~41–49 ГБ/с', 'copy engine и TMA-gather'),
    # prefill routes
    'pre':   (840, 60, 320, 440, 'group', 'Prefill: два маршрута', ''),
    'short': (860, 106, 280, 118, 'lever', 'Короткий кусок  < 2049 новых токенов',
              'резидентные эксперты — D2D внутри GPU (T1); < 160 токенов — только выбранные роутером (C1)'),
    'long':  (860, 244, 280, 118, 'lever', 'Длинный документ  ≥ 2049',
              'layer-major: банк слоя целиком в двойной буфер, копия идёт внахлёст с GEMM (C0)'),
    'dec':   (860, 382, 280, 98, 'lever', 'Decode',
              'промахи: TMA-gather в графе, спрятан за shared-экспертом (T3)'),
    'legend': (840, 560, 320, 224, 'group', 'Обозначения', ''),
    'lg1':   (860, 606, 280, 40, 'lever', 'рычаг этого релиза', ''),
    'lg2':   (860, 656, 280, 40, 'op', 'без изменений', ''),
    'lg3':   (860, 706, 280, 40, 'memhot', 'память, которую мы отдали модели', ''),
}
EDGES = [('tok', 'emb'), ('emb', 'norm1'), ('norm1', 'gdn'), ('norm1', 'attn'), ('gdn', 'norm2'), ('attn', 'norm2'),
         ('norm2', 'router'), ('router', 'lru'), ('lru', 'routed'), ('lru', 'shared'), ('routed', 'epi'),
         ('shared', 'epi'), ('epi', 'head'), ('head', 'out')]
LINKS = [('slots', 'lru', 'попадания'), ('banks', 'pcie', ''), ('pcie', 'buf', 'prefill'), ('pcie', 'slots', 'промахи'),
         ('embh', 'emb', '')]

PALETTE = {
    'light': dict(bg='#f6f7f9', fg='#17202b', muted='#5b6675', line='#9aa6b4', group='#ffffff', groupline='#d3d9e1',
                  op='#ffffff', opline='#9aa6b4', lever='#e3f0f5', leverline='#1f6f8b', mem='#f1f3f6', memline='#b5bfcb',
                  memhot='#e5f3ea', memhotline='#1d7a46', io='#17202b', iofg='#ffffff', link='#fff5e0', linkline='#b7791f'),
    'dark': dict(bg='#0f141a', fg='#e4e9ef', muted='#97a3b2', line='#5c6978', group='#151c24', groupline='#2a3440',
                 op='#1b232d', opline='#5c6978', lever='#16303b', leverline='#5fb3cf', mem='#1b232d', memline='#3a4654',
                 memhot='#17311f', memhotline='#5cc489', io='#e4e9ef', iofg='#0f141a', link='#33290f', linkline='#e0a84a'),
}


def center(b):
    x, y, w, h = BOXES[b][:4]
    return x + w / 2, y + h / 2


def anchor(a, b):
    """Points on the facing edges of boxes a and b."""
    ax, ay, aw, ah = BOXES[a][:4]
    bx, by, bw, bh = BOXES[b][:4]
    (acx, acy), (bcx, bcy) = center(a), center(b)
    if by >= ay + ah - 1:
        return (acx if aw <= bw else bcx if ax <= bcx <= ax + aw else acx, ay + ah), (bcx if bw <= aw else acx if bx <= acx <= bx + bw else bcx, by)
    if ay >= by + bh - 1:
        return (acx, ay), (bcx, by + bh)
    if bx >= ax + aw:
        return (ax + aw, acy), (bx, bcy)
    return (ax, acy), (bx + bw, bcy)


def svg(theme):
    p = PALETTE[theme]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="IBM Plex Sans, Segoe UI, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{p["bg"]}"/>',
           f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{p["line"]}"/></marker></defs>',
           f'<text x="20" y="34" font-size="20" font-weight="600" fill="{p["fg"]}">Qwen3.6-35B-A3B NVFP4 на 16 ГБ: '
           f'как проходит токен и где лежат веса (main128)</text>']
    for bid, (x, y, w, h, kind, title, detail) in BOXES.items():
        if kind == 'group':
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{p["group"]}" '
                       f'stroke="{p["groupline"]}" stroke-dasharray="5 4"/>')
            out.append(f'<text x="{x + 14}" y="{y + 26}" font-size="14" font-weight="600" fill="{p["muted"]}">'
                       f'{html.escape(title)}</text>')
    for a, b in EDGES:
        (x1, y1), (x2, y2) = anchor(a, b)
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{p["line"]}" stroke-width="1.6" marker-end="url(#a)"/>')
    for a, b, label in LINKS:
        (x1, y1), (x2, y2) = anchor(a, b)
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{p["linkline"]}" stroke-width="1.8" '
                   f'stroke-dasharray="6 4" marker-end="url(#a)"/>')
        if label:
            out.append(f'<text x="{(x1 + x2) / 2 + 6}" y="{(y1 + y2) / 2 - 6}" font-size="12" fill="{p["linkline"]}">{label}</text>')
    for bid, (x, y, w, h, kind, title, detail) in BOXES.items():
        if kind == 'group':
            continue
        fill, stroke, fg = {'op': (p['op'], p['opline'], p['fg']), 'lever': (p['lever'], p['leverline'], p['fg']),
                            'mem': (p['mem'], p['memline'], p['fg']), 'memhot': (p['memhot'], p['memhotline'], p['fg']),
                            'io': (p['io'], p['io'], p['iofg']), 'link': (p['link'], p['linkline'], p['fg'])}[kind]
        sw = 2 if kind in ('lever', 'memhot', 'link') else 1.2
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        lines = [title] + wrap(detail, max(18, int(w / 7.2)))
        ty = y + h / 2 - (len(lines) - 1) * 8 + 4
        for i, line in enumerate(lines):
            size, weight, color = (13, 600, fg) if i == 0 else (11.5, 400, fg if kind == 'io' else p['muted'])
            out.append(f'<text x="{x + w / 2}" y="{ty + i * 16}" font-size="{size}" font-weight="{weight}" '
                       f'fill="{color}" text-anchor="middle">{html.escape(line)}</text>')
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def wrap(text, width):
    words, lines, cur = text.split(), [], ''
    for word in words:
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f'{cur} {word}'.strip()
    return lines + ([cur] if cur else [])


def drawio():
    p = PALETTE['light']
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    style = {'group': f'rounded=1;dashed=1;fillColor={p["group"]};strokeColor={p["groupline"]};verticalAlign=top;align=left;spacingLeft=10;fontStyle=1;',
             'op': f'rounded=1;whiteSpace=wrap;fillColor={p["op"]};strokeColor={p["opline"]};',
             'lever': f'rounded=1;whiteSpace=wrap;fillColor={p["lever"]};strokeColor={p["leverline"]};strokeWidth=2;',
             'mem': f'rounded=1;whiteSpace=wrap;fillColor={p["mem"]};strokeColor={p["memline"]};',
             'memhot': f'rounded=1;whiteSpace=wrap;fillColor={p["memhot"]};strokeColor={p["memhotline"]};strokeWidth=2;',
             'io': f'rounded=1;whiteSpace=wrap;fillColor={p["io"]};strokeColor={p["io"]};fontColor={p["iofg"]};',
             'link': f'rounded=1;whiteSpace=wrap;fillColor={p["link"]};strokeColor={p["linkline"]};strokeWidth=2;'}
    order = sorted(BOXES, key=lambda b: BOXES[b][4] != 'group')
    for bid in order:
        x, y, w, h, kind, title, detail = BOXES[bid]
        value = html.escape(f'<b>{title}</b>' + (f'<br><font style="font-size:11px">{detail}</font>' if detail else ''))
        cells.append(f'<mxCell id="{bid}" value="{value}" style="{style[kind]}html=1;" vertex="1" parent="1">'
                     f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, (a, b) in enumerate(EDGES):
        cells.append(f'<mxCell id="e{i}" style="endArrow=block;html=1;strokeColor={p["line"]};" edge="1" parent="1" '
                     f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    for i, (a, b, label) in enumerate(LINKS):
        cells.append(f'<mxCell id="l{i}" value="{label}" style="endArrow=block;dashed=1;html=1;strokeColor={p["linkline"]};'
                     f'fontColor={p["linkline"]};" edge="1" parent="1" source="{a}" target="{b}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    return ('<mxfile host="tools/model-diagram.py"><diagram name="main128">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" page="1" pageWidth="{W}" pageHeight="{H}">'
            '<root>' + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>\n')


def main():
    img = ROOT / 'docs' / 'img'
    img.mkdir(parents=True, exist_ok=True)
    (img / 'model-blocks.svg').write_text(svg('light'), encoding='utf-8', newline='\n')
    (img / 'model-blocks-dark.svg').write_text(svg('dark'), encoding='utf-8', newline='\n')
    (ROOT / 'docs' / 'model-blocks.drawio').write_text(drawio(), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
