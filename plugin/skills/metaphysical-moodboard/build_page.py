"""Build a metaphysical moodboard page: one self-contained HTML file.

    python build_page.py spec.json [--wireframe]

--wireframe writes <spec>.wireframe.html beside the spec with every picture
replaced by a flat grey tile, for checking layout and print without looking
at the person's pictures (blind mode).

The spec names an image folder (usually the board's `board export` folder),
a theme (colours, fonts, image treatment) and a list of spreads. Each spread
is one A4-landscape page in print and one full-width panel on screen. All
positions are in a 297 x 210 grid (millimetres of the printed page), so the
screen and the PDF share one layout.

Spread fields
  id, label          short id, and a small running label (may be "")
  decay              0..1, how far this spread has come apart. It raises
                     blur, lowers contrast and saturation, and fades text.
  ground             optional background colour for this spread
  dark               true for a dark spread (light ink, traces in screen mode)
  items              list of elements, each with x, y, w (mm) and a type

Element types
  image   src (file in the image folder), h (mm, optional, crops by cover),
          fade (0..1 soft dissolving edge), decay (overrides the spread's),
          trace (true draws it as a faint memory of an earlier image),
          focus ("x% y%" object-position)
  text    lines (list of strings). Words in [[double brackets]] are drawn
          half-erased. size (mm), align, ghost (true adds a displaced,
          blurred double), cite (small attribution line), lang,
          opacity (for subliminal text near the threshold of sight),
          tr (translation lines, set smaller right under the original)

Top-level `text_scale` multiplies every text and mark size.
  mark    a tiny label, such as a stage number

Images are resized to `max_side` px and embedded as JPEG data URIs.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sys

from PIL import Image, ImageOps

PAGE_W, PAGE_H = 297.0, 210.0


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def embed(path: str, max_side: int, quality: int) -> tuple[str, float]:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return ("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            im.height / im.width)


def pct_x(v: float) -> str:
    return f"{v / PAGE_W * 100:.3f}%"


def pct_y(v: float) -> str:
    return f"{v / PAGE_H * 100:.3f}%"


def cq(v_mm: float) -> str:
    # one mm of the printed page, as a fraction of the spread's width
    return f"{v_mm / PAGE_W * 100:.4f}cqw"


def words(line: str) -> str:
    out, pos = [], 0
    for m in re.finditer(r"\[\[(.+?)\]\]", line):
        out.append(esc(line[pos:m.start()]))
        out.append(f'<span class="erased">{esc(m.group(1))}</span>')
        pos = m.end()
    out.append(esc(line[pos:]))
    return "".join(out)


def plain(line: str) -> str:
    return re.sub(r"\[\[(.+?)\]\]", r"\1", line)


def build(spec: dict, spec_dir: str) -> str:
    theme = spec.get("theme", {})
    img_dir = os.path.join(spec_dir, spec["images"]) if not os.path.isabs(spec["images"]) else spec["images"]
    max_side = int(spec.get("max_side", 1500))
    quality = int(spec.get("quality", 78))
    cache: dict[str, tuple[str, float]] = {}

    wire = bool(spec.get("_wireframe"))
    text_scale = float(spec.get("text_scale", 1.0))

    def img(src: str) -> tuple[str, float]:
        if wire:
            # flat tiles for checking layout without looking at the pictures
            tone = 150 + (sum(map(ord, src)) % 60)
            buf = io.BytesIO()
            Image.new("RGB", (30, 20), (tone, tone, tone)).save(buf, "PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), 2 / 3
        if src not in cache:
            cache[src] = embed(os.path.join(img_dir, src), max_side, quality)
        return cache[src]

    spreads_html = []
    for n, sp in enumerate(spec["spreads"]):
        d0 = float(sp.get("decay", 0))
        style = f' style="--ground:{sp["ground"]}"' if sp.get("ground") else ""
        cls = "spread dark" if sp.get("dark") else "spread"
        parts = [f'<section class="{cls}" id="{esc(sp["id"])}"{style}>']
        for it in sp.get("items", []):
            x, y, w = it["x"], it["y"], it["w"]
            box = f"left:{pct_x(x)};top:{pct_y(y)};width:{pct_x(w)};"
            kind = it.get("type", "image")
            if kind == "image":
                uri, ratio = img(it["src"])
                h = it.get("h", w * ratio)
                d = float(it.get("decay", d0))
                fade = float(it.get("fade", 0.6 if it.get("trace") else 0))
                cls = "im trace" if it.get("trace") else "im"
                vars_ = (f"--d:{d:.3f};--fade:{fade:.3f};"
                         + (f"--o:{it['opacity']};" if "opacity" in it else ""))
                focus = f' style="object-position:{it["focus"]}"' if it.get("focus") else ""
                parts.append(
                    f'<figure class="{cls}" style="{box}height:{pct_y(h)};{vars_}">'
                    f'<img src="{uri}" alt="{esc(it.get("alt", ""))}"{focus}></figure>')
            elif kind == "text":
                d = float(it.get("decay", d0))
                size = float(it.get("size", 4.2)) * text_scale
                align = it.get("align", "left")
                lang = f' lang="{esc(it["lang"])}"' if it.get("lang") else ""
                lines = it["lines"]
                body = "<br>".join(words(l) for l in lines)
                ghost = ""
                if it.get("ghost"):
                    ghost = (f'<span class="ghost" aria-hidden="true">'
                             f'{"<br>".join(esc(plain(l)) for l in lines)}</span>')
                tr = ""
                if it.get("tr"):
                    tr = '<p class="tr">' + "<br>".join(words(l) for l in it["tr"]) + "</p>"
                cite = f'<span class="cite">{esc(it["cite"])}</span>' if it.get("cite") else ""
                cite = tr + cite
                cls = "tx " + it.get("cls", "")
                parts.append(
                    f'<div class="{cls.strip()}"{lang} style="{box}--d:{d:.3f};'
                    + (f'opacity:{it["opacity"]};' if "opacity" in it else "") +
                    f'font-size:{cq(size)};text-align:{align};">'
                    f'<p>{ghost}<span class="line">{body}</span></p>{cite}</div>')
            elif kind == "mark":
                size = float(it.get("size", 2.4)) * text_scale
                parts.append(f'<div class="mk" style="{box}font-size:{cq(size)};">'
                             f'{esc(it["text"])}</div>')
        if sp.get("label"):
            parts.append(f'<div class="run">{esc(sp["label"])}</div>')
        parts.append("</section>")
        spreads_html.append("".join(parts))

    t = {
        "paper": theme.get("paper", "#ebe9e4"),
        "ink": theme.get("ink", "#3b3935"),
        "ink_faint": theme.get("ink_faint", "#8f8b84"),
        "serif": theme.get("serif", '"Cormorant Garamond", "EB Garamond", Garamond, "Times New Roman", serif'),
        "mono": theme.get("mono", '"IBM Plex Mono", "Courier New", monospace'),
        "dark_paper": theme.get("dark_paper", "#16130f"),
        "dark_ink": theme.get("dark_ink", "#b9b3a8"),
    }
    css = CSS
    for k, v in t.items():
        css = css.replace("{{" + k + "}}", v)
    return (
        '<!doctype html><html lang="' + esc(spec.get("lang", "en")) + '"'
        + (f' data-theme="{esc(theme["force"])}"' if theme.get("force") else "")
        + '><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(spec["title"])}</title>'
        f'<meta name="description" content="{esc(spec.get("description", ""))}">'
        f"<style>{css}</style></head><body>"
        + "".join(spreads_html)
        + f"<script>{JS}</script></body></html>"
    )


CSS = r"""
:root{--paper:{{paper}};--ink:{{ink}};--ink-faint:{{ink_faint}};
 --serif:{{serif}};--mono:{{mono}}}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:{{dark_paper}};--ink:{{dark_ink}};--ink-faint:#6d685f}}
:root[data-theme="dark"]{--paper:{{dark_paper}};--ink:{{dark_ink}};--ink-faint:#6d685f}
*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--paper)}
body{background:var(--paper);color:var(--ink);font-family:var(--serif);
 padding:clamp(40px,6vh,90px) 16px clamp(120px,20vh,260px);display:flex;flex-direction:column;
 align-items:center;gap:clamp(90px,16vh,220px);
 -webkit-font-smoothing:antialiased}
.spread{position:relative;width:min(100%,1400px);aspect-ratio:297/210;container-type:inline-size;
 background:var(--ground,transparent);overflow:hidden}
.im{position:absolute;overflow:hidden;
 opacity:var(--o,calc(1 - var(--d)*0.45));
 filter:grayscale(calc(0.35 + var(--d)*0.6)) contrast(calc(1 - var(--d)*0.35))
  brightness(calc(1 + var(--d)*0.12)) blur(calc(var(--d)*var(--d)*0.55cqw));
 transition:filter 6s ease, opacity 6s ease;
 --e:calc(var(--fade)*30%);
 -webkit-mask-image:linear-gradient(90deg,transparent,#000 var(--e),#000 calc(100% - var(--e)),transparent),
  linear-gradient(180deg,transparent,#000 var(--e),#000 calc(100% - var(--e)),transparent);
 -webkit-mask-composite:source-in;
 mask-image:linear-gradient(90deg,transparent,#000 var(--e),#000 calc(100% - var(--e)),transparent),
  linear-gradient(180deg,transparent,#000 var(--e),#000 calc(100% - var(--e)),transparent);
 mask-composite:intersect}
.im img{display:block;width:100%;height:100%;object-fit:cover}
.im:hover{filter:grayscale(calc(0.2 + var(--d)*0.4)) contrast(1) blur(0);opacity:1;transition:filter 2.5s ease,opacity 2.5s ease}
.trace{mix-blend-mode:multiply;pointer-events:none;
 opacity:var(--o,.12);filter:grayscale(1) contrast(.7) blur(.35cqw)}
.tx{position:absolute;line-height:1.38;letter-spacing:.01em;
 color:color-mix(in srgb,var(--ink) calc(100% - var(--d)*55%),transparent)}
.tx p{position:relative}
.tx .ghost{position:absolute;left:0;top:0;transform:translate(.35em,-.42em);
 color:var(--ink);opacity:.13;filter:blur(.06em);pointer-events:none;width:100%}
.tx .erased{opacity:.22;filter:blur(.035em);transition:opacity 4s ease,filter 4s ease}
.tx:hover .erased{opacity:.55;filter:none}
.tx .tr{margin-top:.7em;font-size:.74em;opacity:.78}
.tx .cite{display:block;margin-top:1em;font-family:var(--mono);font-size:.58em;letter-spacing:.1em;
 text-transform:uppercase;color:var(--ink-faint)}
.tx.small{font-style:italic}
.tx.title{letter-spacing:.32em;text-transform:lowercase;font-weight:300}
.mk{position:absolute;font-family:var(--mono);letter-spacing:.2em;color:var(--ink-faint);text-transform:uppercase}
.run{position:absolute;right:3.4%;bottom:4%;font-family:var(--mono);font-size:.95cqw;letter-spacing:.24em;
 color:var(--ink-faint);text-transform:uppercase;opacity:.7}
.dark{--ink:#b9b3a8;--ink-faint:#6d685f;color:var(--ink)}
.dark .trace{mix-blend-mode:screen}
@media (prefers-reduced-motion:reduce){.im,.tx .erased{transition:none}}
@media print{
 @page{size:297mm 210mm;margin:0}
 html,body{background:var(--paper);-webkit-print-color-adjust:exact;print-color-adjust:exact}
 body{padding:0;gap:0;display:block}
 .spread{width:297mm;height:210mm;aspect-ratio:auto;break-after:page;break-inside:avoid;page-break-after:always}
 .spread:last-child{break-after:auto;page-break-after:auto}
 .im,.tx .erased{transition:none}
}
@media (max-width:520px){body{gap:10vh}}
"""

# Slow drift of the traces: a memory that will not stay still. Off for
# reduced motion and print, harmless without JS.
JS = r"""
(function(){
 if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
 var tr = Array.prototype.slice.call(document.querySelectorAll('.trace'));
 var t0 = performance.now();
 function step(t){
  var s = (t - t0) / 1000;
  tr.forEach(function(el, i){
   var a = Math.sin(s / (17 + i * 3) + i) * 0.5 + 0.5;
   el.style.transform = 'translate(' + (Math.sin(s / (23 + i)) * 0.4).toFixed(3) + '%,' +
     (Math.cos(s / (29 + i)) * 0.4).toFixed(3) + '%)';
   el.style.opacity = (0.06 + a * 0.10).toFixed(3);
  });
  requestAnimationFrame(step);
 }
 requestAnimationFrame(step);
 addEventListener('beforeprint', function(){ tr.forEach(function(el){ el.style.transform=''; el.style.opacity=''; }); });
})();
"""


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    spec_path = os.path.abspath(sys.argv[1])
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    wire = "--wireframe" in sys.argv
    spec["_wireframe"] = wire
    page = build(spec, os.path.dirname(spec_path))
    out = spec.get("out") or os.path.splitext(spec_path)[0] + ".page.html"
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(spec_path), out)
    if wire:
        out = os.path.splitext(spec_path)[0] + ".wireframe.html"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {out} ({len(page) / 1e6:.1f} MB, {len(spec['spreads'])} spreads)")


if __name__ == "__main__":
    main()
