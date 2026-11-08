# compose.py -- a board as one self-contained HTML page, or as one picture
#
# A board is the deliverable of a moodboard, so it has to stand on its own
# away from Arcana: sent to someone, opened on another machine, printed. The
# HTML here is one file. Every picture is embedded as a JPEG data URI, so the
# page opens where the original paths do not exist, and each also names its
# original by file:// path, which the lightbox opens when it is there.
#
# Three layouts share one markup, as a class on the root element:
#
#   editorial   the hero across the top, then each section's pictures in a
#               ragged grid sized by role (supports large, details small)
#   masonry     columns of pictures at their own proportions
#   grid        even square tiles, for comparing like with like
#
# The PNG is the same board drawn with PIL, for an agent in pixels mode to look
# at its own layout. Colours and type come from ui_style.py, so a board and the
# app agree.

from __future__ import annotations

import base64
import html
import io
import os
import time
from urllib.parse import quote

from PIL import Image, ImageDraw

try:
    from . import sheets as _sheets
    from . import ui_style as _ui
except ImportError:                                   # pragma: no cover
    import sheets as _sheets
    import ui_style as _ui

LAYOUTS = ("editorial", "masonry", "grid")
HERO_SIDE = 1400
ITEM_SIDE = 640

# The share of a row a picture starts from in the editorial layout, by its role.
# Rows are flex rows: pictures grow from these shares until the row is full, so
# a support stays about twice a detail and no row is left with a hole in it.
SPAN = {"hero": 100, "support": 30, "counterpoint": 30, "texture": 22, "detail": 14, None: 30}


def _esc(s) -> str:
    return html.escape(str(s or ""), quote=True)


def _data_uri(path: str, side: int) -> str | None:
    jpeg = _sheets.fitted_jpeg(path, side, quality=82)
    if not jpeg:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _file_url(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    return "file:///" + quote(p.lstrip("/"), safe="/:")


def _figure(item, side: int, extra_class: str = "") -> str:
    uri = _data_uri(item.path, side) if not item.missing else None
    role = item.role or ""
    kind = item.kind if item.kind in ("derived", "generated") else ""
    span = SPAN.get(item.role, 4)
    classes = " ".join(c for c in ("item", f"role-{role}" if role else "", f"kind-{kind}" if kind else "",
                                   extra_class) if c)
    if uri:
        media = (f'<img src="{uri}" data-full="{_esc(_file_url(item.path))}" '
                 f'alt="{_esc(item.handle or item.file)}" loading="lazy">')
    else:
        media = f'<div class="missing">file missing<br><small>{_esc(item.file)}</small></div>'
    chips = []
    if role:
        chips.append(f'<span class="chip role">{_esc(role)}</span>')
    if kind:
        made = f" from {_esc(item.derived_from)}" if item.derived_from else ""
        chips.append(f'<span class="chip made">{kind.upper()}{made}</span>')
    note = f'<span class="note">{_esc(item.note)}</span>' if item.note else ""
    return (f'<figure class="{classes}" style="--span:{span}">{media}'
            f'<figcaption><span class="num">{item.num:02d}</span>{"".join(chips)}{note}'
            f'<span class="handle">{_esc(item.handle or item.file)}</span></figcaption></figure>')


def _palette(board) -> str:
    if not board.palette:
        return ""
    total = sum((p.weight or 1.0) for p in board.palette) or 1.0
    cells = "".join(
        f'<div class="swatch" style="flex:{(p.weight or 1.0) / total:.3f};background:{_esc(p.hex)}"></div>'
        for p in board.palette)
    # The names go in a legend under the strip: a narrow swatch has no room for one.
    legend = "".join(
        f'<span><i style="background:{_esc(p.hex)}"></i>{_esc(p.hex)}'
        + (f" {_esc(p.role)}" if p.role else "")
        + (f" {(p.weight or 0) / total:.0%}" if p.weight else "") + "</span>"
        for p in board.palette)
    return (f'<div class="palette" aria-label="palette">{cells}</div>'
            f'<div class="legend">{legend}</div>')


CSS = """
:root{--bg:%(BG)s;--surface:%(SURFACE)s;--line:%(LINE)s;--ink:%(INK)s;--dim:%(INK_DIM)s;
--faint:%(INK_FAINT)s;--accent:%(ACCENT)s;--warn:%(WARN)s;--gap:14px}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif;
background:var(--bg)}
main{max-width:1500px;margin:0 auto;padding:48px 40px 64px}
.kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim)}
h1{font-size:clamp(30px,4.2vw,56px);line-height:1.05;margin:.25em 0 .35em;font-weight:600;letter-spacing:-.01em}
.brief{color:var(--dim);font-size:14px;margin:0 0 .6em}
.concept{font-size:clamp(17px,1.5vw,21px);line-height:1.5;max-width:52em;margin:0 0 1.4em}
.palette{display:flex;height:44px;border-radius:6px;overflow:hidden;margin:0;border:1px solid var(--line)}
.swatch{min-width:6px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:10px 0 40px;font-size:11px;color:var(--dim);
letter-spacing:.04em}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
vertical-align:-1px;border:1px solid var(--line)}
.angle{margin:0 0 56px}
.angle>header{display:grid;grid-template-columns:auto 1fr;column-gap:16px;align-items:baseline;
border-top:1px solid var(--line);padding-top:14px;margin-bottom:18px}
.angle .snum{font-size:12px;color:var(--accent);letter-spacing:.12em;font-variant-numeric:tabular-nums}
.angle h2{margin:0;font-size:22px;font-weight:600}
.angle .why{grid-column:2;margin:.3em 0 0;color:var(--dim);max-width:46em}
.angle .queries{grid-column:2;margin:.35em 0 0;color:var(--faint);font-size:12px}
figure{margin:0;position:relative}
figure img{display:block;width:100%%;height:auto;border-radius:4px;cursor:zoom-in;background:var(--surface)}
figcaption{display:flex;flex-wrap:wrap;gap:6px;align-items:baseline;font-size:12px;color:var(--dim);
padding:7px 2px 0}
.num{color:var(--faint);font-variant-numeric:tabular-nums}
.chip{font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:1px 6px;border-radius:3px;
border:1px solid var(--line);color:var(--ink)}
.chip.made{background:var(--warn);color:#2a1409;border-color:var(--warn)}
.note{flex-basis:100%%;color:var(--ink);font-size:13px;line-height:1.45}
.handle{margin-left:auto;color:var(--faint);font-size:11px}
.missing{aspect-ratio:3/2;display:grid;place-items:center;text-align:center;color:var(--faint);
background:var(--surface);border:1px dashed var(--line);border-radius:4px;font-size:12px}
.kind-derived img,.kind-generated img{outline:2px solid var(--warn);outline-offset:-2px}
/* editorial */
.layout-editorial .hero{margin:0 0 56px}
.layout-editorial .hero img{max-height:80vh;width:auto;max-width:100%%;margin:0 auto}
.layout-editorial .items{display:flex;flex-wrap:wrap;gap:var(--gap);align-items:flex-start}
.layout-editorial .items figure{flex:var(--span) 1 calc(var(--span) * 1%% - var(--gap));min-width:150px}
.layout-editorial .role-counterpoint{padding:8px;border:1px solid var(--accent);border-radius:6px}
/* masonry */
.layout-masonry .items{columns:3 300px;column-gap:var(--gap)}
.layout-masonry .items figure{break-inside:avoid;margin-bottom:var(--gap)}
/* grid */
.layout-grid .items{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:var(--gap)}
.layout-grid .items img{aspect-ratio:1;object-fit:cover}
footer{border-top:1px solid var(--line);padding-top:14px;color:var(--faint);font-size:12px}
.lb{position:fixed;inset:0;background:rgba(10,9,8,.94);display:grid;place-items:center;z-index:9;padding:24px}
.lb[hidden]{display:none}
.lb img{max-width:96vw;max-height:86vh;object-fit:contain}
.lb p{color:var(--dim);font-size:13px;margin:12px 0 0;text-align:center}
@media (max-width:760px){main{padding:28px 16px}.layout-editorial .items figure{flex-basis:45%%}}
@media print{html,body{background:#fff;color:#111}.lb{display:none}figure,.angle>header{break-inside:avoid}
.chip{color:#111}.dim,.why,figcaption{color:#444}}
"""

SCRIPT = """
(function(){var figs=[].slice.call(document.querySelectorAll('figure.item img'));if(!figs.length)return;
var box=document.getElementById('lb'),im=box.querySelector('img'),cap=box.querySelector('p'),i=-1;
function show(k){i=(k+figs.length)%figs.length;var f=figs[i];im.onerror=function(){im.onerror=null;im.src=f.src};
im.src=f.getAttribute('data-full')||f.src;var c=f.closest('figure').querySelector('figcaption');
cap.textContent=c?c.innerText.replace(/\\s+/g,' '):'';box.hidden=false}
figs.forEach(function(f,k){f.addEventListener('click',function(){show(k)})});
box.addEventListener('click',function(){box.hidden=true});
document.addEventListener('keydown',function(e){if(box.hidden)return;if(e.key==='Escape')box.hidden=true;
if(e.key==='ArrowRight')show(i+1);if(e.key==='ArrowLeft')show(i-1)})})();
"""


def to_html(board, layout: str = "editorial") -> str:
    """The board as one self-contained HTML page."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout is one of {', '.join(LAYOUTS)}, got {layout!r}.")
    items = board.items
    hero = next((i for i in items if i.role == "hero"), None)
    m = board.metrics
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             f"<title>{_esc(board.name)}</title><style>",
             CSS % {k: getattr(_ui, k) for k in ("BG", "SURFACE", "LINE", "INK", "INK_DIM",
                                                  "INK_FAINT", "ACCENT", "WARN")},
             f'</style></head><body class="layout-{layout}"><main>',
             '<header class="masthead">',
             f'<div class="kicker">Moodboard · {len(items)} picture{"s" if len(items) != 1 else ""}'
             f'{" · " + _esc(board.dataset) if board.dataset else ""} · '
             f'{time.strftime("%d %B %Y")}</div>',
             f"<h1>{_esc(board.name)}</h1>"]
    if board.brief:
        parts.append(f'<p class="brief">Brief: {_esc(board.brief)}</p>')
    if board.concept:
        parts.append(f'<p class="concept">{_esc(board.concept)}</p>')
    parts.append(_palette(board))
    parts.append("</header>")

    if hero is not None and layout == "editorial":
        parts.append(_figure(hero, HERO_SIDE, "hero"))

    def section(sid, title, why, queries, members, n):
        # The editorial layout has already drawn the hero across the top.
        shown = [i for i in members if not (layout == "editorial" and i is hero)]
        if not shown:
            return ""
        head = (f'<header><span class="snum">{n:02d}</span><h2>{_esc(title)}</h2>'
                + (f'<p class="why">{_esc(why)}</p>' if why else "")
                + (f'<p class="queries">Searched as: {_esc("; ".join(queries))}</p>' if queries else "")
                + "</header>")
        figs = "".join(_figure(i, ITEM_SIDE) for i in shown)
        return f'<section class="angle" id="{_esc(sid)}">{head}<div class="items">{figs}</div></section>'

    n = 0
    for s in board.sections:
        n += 1
        parts.append(section(s.id, s.title, s.rationale, s.queries, s.items, n))
    if board.unsectioned:
        n += 1
        title = "Also on the board" if board.sections else "The pictures"
        parts.append(section("loose", title, "", [], board.unsectioned, n))

    bits = [f"{m.count} pictures"]
    if m.diversity is not None:
        bits.append(f"diversity {m.diversity:.2f}")
    if m.derived or m.generated:
        bits.append(f"{m.derived + m.generated} derived or generated, marked in ochre")
    bits.append("near-duplicates " + ("none" if not m.near_duplicates else str(len(m.near_duplicates))))
    parts.append(f'<footer>{" · ".join(_esc(b) for b in bits)} · made with Arcana from '
                 f'{_esc(board.dataset or "the archive")}. Click a picture to see it whole.</footer>')
    parts.append('</main><div class="lb" id="lb" hidden><div><img alt=""><p></p></div></div>')
    parts.append(f"<script>{SCRIPT}</script></body></html>")
    return "".join(p for p in parts if p)


# ─────────────────────────── the board as one picture ───────────────────────────

def _rgb(hexcode: str) -> tuple:
    h = hexcode.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _wrap(draw, text: str, font, width: int) -> list[str]:
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def _rows(images: list, width: int, target: int, gap: int):
    """Justified rows: each row's pictures scaled to one height that fills the width."""
    rows, cur = [], []
    for im in images:
        cur.append(im)
        ratio = sum(i.width / i.height for i in cur)
        if ratio * target + gap * (len(cur) - 1) >= width:
            h = (width - gap * (len(cur) - 1)) / ratio
            rows.append((cur, int(h)))
            cur = []
    if cur:
        # A short last row grows (up to half again) toward the full width, so a
        # section of two pictures does not leave half the page empty.
        ratio = sum(i.width / i.height for i in cur)
        fit = (width - gap * (len(cur) - 1)) / ratio
        rows.append((cur, int(min(fit, target * 1.5))))
    return rows


def to_png(board, width: int = 1568) -> Image.Image:
    """
    The board drawn as one picture, at the long edge a model sees well (A5).

    Each picture is numbered as `board show` numbers it, so a critique can name
    pictures, and a derived one is framed in ochre.
    """
    pad, gap = 40, 10
    inner = width - 2 * pad
    bg, ink, dim, line, warn = (_rgb(_ui.BG), _rgb(_ui.INK), _rgb(_ui.INK_DIM),
                                _rgb(_ui.LINE), _rgb(_ui.WARN))
    f_title, f_text, f_small = _sheets._font(40), _sheets._font(17), _sheets._font(13)
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    blocks = []        # ("text", lines, font, colour) | ("rule",) | ("palette",) | ("rows", rows)

    def text(s, font, colour, w=inner):
        blocks.append(("text", _wrap(scratch, s, font, w), font, colour))

    text(board.name, f_title, ink)
    if board.concept:
        text(board.concept, f_text, dim)
    if board.palette:
        blocks.append(("palette",))

    def load(item, side):
        im = _sheets._open_fitted(item.path, side) if not item.missing else None
        if im is None:
            im = Image.new("RGB", (side, int(side * 2 / 3)), _rgb(_ui.SURFACE))
        return (im, item)

    hero = next((i for i in board.items if i.role == "hero"), None)
    if hero is not None:
        im, _ = load(hero, 1200)
        scale = min(inner / im.width, 620 / im.height)
        blocks.append(("rows", [([(im.resize((int(im.width * scale), int(im.height * scale))), hero)],
                                 int(im.height * scale))]))
    groups = [(s.title, s.rationale, [i for i in s.items if i is not hero]) for s in board.sections]
    loose = [i for i in board.unsectioned if i is not hero]
    if loose:
        groups.append(("Also on the board" if board.sections else "", "", loose))
    for title, why, members in groups:
        if not members:
            continue
        blocks.append(("rule",))
        if title:
            text(title, f_text, ink)
        if why:
            text(why, f_small, dim)
        loaded = [load(i, 520) for i in members]
        rows = _rows([im for im, _ in loaded], inner, 230, gap)
        k, packed = 0, []
        for row, h in rows:
            packed.append(([(im.resize((max(1, int(im.width * h / im.height)), h)), loaded[k + j][1])
                            for j, im in enumerate(row)], h))
            k += len(row)
        blocks.append(("rows", packed))

    def height_of(b):
        if b[0] == "text":
            return len(b[1]) * (b[2].size + 8) + 6
        if b[0] == "rule":
            return 22
        if b[0] == "palette":
            return 46
        return sum(h + gap for _, h in b[1])

    total = pad * 2 + sum(height_of(b) for b in blocks)
    canvas = Image.new("RGB", (width, total), bg)
    d = ImageDraw.Draw(canvas)
    y = pad
    for b in blocks:
        if b[0] == "text":
            for ln in b[1]:
                d.text((pad, y), ln, font=b[2], fill=b[3])
                y += b[2].size + 8
            y += 6
        elif b[0] == "rule":
            d.line([(pad, y + 10), (width - pad, y + 10)], fill=line, width=1)
            y += 22
        elif b[0] == "palette":
            weights = [(p.weight or 1.0) for p in board.palette]
            x, tw = pad, sum(weights) or 1.0
            for p, w in zip(board.palette, weights):
                wpx = int(inner * w / tw)
                d.rectangle([x, y, x + wpx, y + 32], fill=_rgb(p.hex))
                x += wpx
            y += 46
        else:
            for row, h in b[1]:
                used = sum(im.width for im, _ in row) + gap * (len(row) - 1)
                x = pad + max(0, (inner - used) // 2) if len(b[1]) == 1 and len(row) == 1 else pad
                for im, item in row:
                    canvas.paste(im, (x, y))
                    if item.kind in ("derived", "generated"):
                        d.rectangle([x, y, x + im.width - 1, y + h - 1], outline=warn, width=3)
                    label = f"{item.num}" + (f" {item.role}" if item.role else "")
                    tw = int(d.textlength(label, font=f_small)) + 10
                    d.rectangle([x, y, x + tw, y + 20], fill=(0, 0, 0))
                    d.text((x + 5, y + 3), label, font=f_small, fill=(255, 255, 255))
                    x += im.width + gap
                y += h + gap
    return canvas


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
