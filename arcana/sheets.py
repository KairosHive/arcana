# sheets.py — many pictures as one image, for a model to look at
#
# A language model pays for an image by its area: roughly width x height / 750
# tokens, however many pictures are packed into it. So the cost of showing an
# agent forty candidates is set by how large each one is drawn, not by how many
# there are, and the useful lever is tile size per purpose:
#
#   160 px   scanning candidates, surveying a cluster       ~35 tokens a picture
#   320 px   comparing finalists                            ~135
#   1024 px  looking closely at one picture                 ~1,400
#
# A grid is also what curation needs: pictures side by side, each with a number
# the agent can refer to without spelling out a path.

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont, ImageOps

BACKGROUND = (18, 18, 20)
MISSING = (60, 60, 64)
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 255, 255)
TITLE_FG = (230, 230, 230)
GAP = 4

# Decoding is most of the cost and Pillow releases the GIL while it decodes.
_DECODE_WORKERS = 8


def _font(size: int):
    """Pillow's bundled font at a given size, or its bitmap font on old versions."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _open_fitted(path: str, side: int) -> Image.Image | None:
    """
    The picture, upright, scaled to fit a side x side box. None if unreadable.

    draft() asks the JPEG decoder for a reduced-scale decode. A 24-megapixel
    photograph shrunk to 160 px never needs its full resolution in memory, and
    decoding at 1/8 scale is several times faster than decoding and resizing.
    """
    try:
        with Image.open(path) as im:
            im.draft("RGB", (side * 2, side * 2))
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((side, side), Image.LANCZOS)
            return im
    except Exception:
        return None


def fitted_jpeg(path: str, max_side: int = 1024, quality: int = 85) -> bytes | None:
    """One picture as a JPEG no larger than max_side on its long edge."""
    im = _open_fitted(path, max_side)
    if im is None:
        return None
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def contact_sheet(paths: list[str], numbers: list[int] | None = None, *,
                  tile: int = 160, cols: int = 6, title: str | None = None,
                  quality: int = 80) -> bytes:
    """
    A JPEG grid of `paths`, each tile labelled with its number.

    A file that can't be read gets a grey tile rather than disappearing, so the
    numbers still line up with the table that accompanies the sheet.
    """
    numbers = list(numbers) if numbers is not None else list(range(1, len(paths) + 1))
    n = len(paths)
    cols = max(1, min(cols, n)) if n else 1
    rows = (n + cols - 1) // cols if n else 1

    title_h = 0
    title_font = _font(max(12, tile // 9))
    if title:
        title_h = max(18, tile // 7) + GAP

    width = cols * tile + (cols + 1) * GAP
    height = title_h + rows * tile + (rows + 1) * GAP
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    if title:
        draw.text((GAP + 2, GAP), title, fill=TITLE_FG, font=title_font)

    with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as pool:
        thumbs = list(pool.map(lambda p: _open_fitted(p, tile), paths))

    label_font = _font(max(11, tile // 11))
    for i, (thumb, num) in enumerate(zip(thumbs, numbers)):
        r, c = divmod(i, cols)
        x0 = GAP + c * (tile + GAP)
        y0 = title_h + GAP + r * (tile + GAP)
        if thumb is None:
            draw.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], fill=MISSING)
            draw.text((x0 + 6, y0 + tile // 2 - 6), "missing", fill=LABEL_FG, font=label_font)
        else:
            ox = x0 + (tile - thumb.width) // 2
            oy = y0 + (tile - thumb.height) // 2
            sheet.paste(thumb, (ox, oy))

        label = str(num)
        box = draw.textbbox((0, 0), label, font=label_font)
        lw, lh = box[2] - box[0], box[3] - box[1]
        draw.rectangle([x0, y0, x0 + lw + 8, y0 + lh + 8], fill=LABEL_BG)
        draw.text((x0 + 4 - box[0], y0 + 4 - box[1]), label, fill=LABEL_FG, font=label_font)

    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=quality)
    return buf.getvalue()
