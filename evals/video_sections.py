"""
Run the video reader and the section detector on real footage, with real CLIP
ticks, and say how long each step took. See VIDEO_ROADMAP.md, 2.9.

    uv run python evals/video_sections.py <folder or file> [...] --n 6 --out <dir>

For each video it prints the probe, the timings and the sections of every
level. With --out it also writes one contact sheet per video: a keyframe for
each coarse section over its time range, and a strip of the boundary strengths,
which is how the sectioning is judged by eye.
"""

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import modality, motion as M, sections as S, video as V, video_index as VI  # noqa: E402

MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def find(paths, n, seed, max_minutes):
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
            continue
        found = []
        for folder, _, names in os.walk(p):
            found += [os.path.join(folder, f) for f in names
                      if os.path.splitext(f)[1].lower() in modality.VIDEO_EXTS]
        random.Random(seed).shuffle(found)
        kept = []
        for f in found:
            if len(kept) >= n:
                break
            try:
                if V.probe(f)["duration"] <= max_minutes * 60:
                    kept.append(f)
            except V.VideoError:
                pass
        files += kept
    return files


def clip_vectors(frames, batch=64):
    import torch
    from arcana import db
    model, proc = db.load_clip(model_id=MODEL)
    out = []
    for i in range(0, len(frames), batch):
        px = proc(images=frames[i:i + batch], return_tensors="pt").pixel_values.to(model.device)
        px = px.to(next(model.parameters()).dtype)
        with torch.no_grad():
            out.append(model.get_image_features(px).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def sheet(path, info, result, out_dir):
    from PIL import Image, ImageDraw
    coarse = result["levels"]["coarse"]
    tile, pad = 200, 6
    shown = coarse[:12]
    w = max(len(shown), 4) * (tile + pad) + pad
    im = Image.new("RGB", (w, tile + 120), (24, 22, 20))
    d = ImageDraw.Draw(im)
    for k, s in enumerate(shown):
        f = V.frame_at(path, (s.t0 + s.t1) / 2, side=tile, info=info)
        if f is not None:
            im.paste(Image.fromarray(f), (pad + k * (tile + pad), pad))
        d.text((pad + k * (tile + pad), tile + pad + 2), f"{s.t0:.1f}-{s.t1:.1f}s", fill=(230, 220, 200))
    # Boundary strengths along the video, with each level's boundaries marked.
    s = result["strengths"]
    y0, h = tile + 40, 60
    if len(s):
        top = float(s.max()) or 1.0
        for i, v in enumerate(s):
            x = pad + int((i + 1) / (len(s) + 1) * (w - 2 * pad))
            d.line([(x, y0 + h), (x, y0 + h - int(h * v / top))], fill=(120, 110, 100))
        for name, colour in (("fine", (90, 120, 160)), ("medium", (200, 160, 70)), ("coarse", (220, 90, 60))):
            for sec in result["levels"][name][1:]:
                x = pad + int(sec.i0 / (len(s) + 1) * (w - 2 * pad))
                d.line([(x, y0 - 4), (x, y0 + h)], fill=colour)
    d.text((pad, y0 + h + 4), f"{os.path.basename(path)}  {info['duration']:.1f}s  "
           f"coarse {len(coarse)} / medium {len(result['levels']['medium'])} / fine {len(result['levels']['fine'])}",
           fill=(230, 220, 200))
    name = os.path.splitext(os.path.basename(path))[0][:60] + ".jpg"
    im.save(os.path.join(out_dir, name), quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--n", type=int, default=6, help="videos to take from each folder")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=6.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    files = find(a.paths, a.n, a.seed, a.max_minutes)
    print(f"{len(files)} videos")
    totals = {"seconds": 0.0, "decode": 0.0, "encode": 0.0, "section": 0.0, "ticks": 0}
    for path in files:
        try:
            spent = {"encode": 0.0}

            def encode(fr):
                t0 = time.perf_counter()
                out = S.unit_rows(clip_vectors(fr))
                spent["encode"] += time.perf_counter() - t0
                return out

            t = time.perf_counter()
            got = VI.analyse(path, encode)
            t_all = time.perf_counter() - t
            if got is None:
                print(f"\nNO FRAMES {path}")
                continue
            info, X, r, cuts = got["info"], got["X"], got["result"], got["cuts"]
            frames = X
            t_encode = spent["encode"]
            t = time.perf_counter()
            S.section_ticks(X, cuts=cuts, duration=info["duration"])
            t_section = time.perf_counter() - t
            t_decode = t_all - t_encode - t_section
        except Exception as e:                                    # noqa: BLE001
            print(f"\nFAILED {path}: {type(e).__name__}: {e}")
            continue
        totals["seconds"] += info["duration"]; totals["decode"] += t_decode
        totals["encode"] += t_encode; totals["section"] += t_section; totals["ticks"] += len(frames)
        top = float(r["strengths"].max()) if len(r["strengths"]) else 0.0
        print(f"\n{os.path.basename(path)}")
        print(f"  {info['duration']:.1f}s  {info['width']}x{info['height']}  {info['fps']:.2f} fps  "
              f"sound {'yes' if info['has_audio'] else 'no'}  ticks {len(frames)}  cuts {len(cuts)}")
        print(f"  read and flow {t_decode:.2f}s  encode {t_encode:.2f}s  section {t_section * 1000:.1f}ms  "
              f"strongest boundary {top:.4f}")
        for s in r["levels"]["coarse"][:6]:
            print(f"    {s.t0:6.1f}-{s.t1:<6.1f} {M.describe(VI.section_motion(got['moves'], s))}")
        for name, _ in S.LEVELS:
            made = r["levels"].get(name, [])
            spans = "  ".join(f"{s.t0:.1f}-{s.t1:.1f}" for s in made[:14])
            print(f"  {name:<7}{len(made):>3}  {spans}{'  ...' if len(made) > 14 else ''}")
        if a.out and len(frames):
            sheet(path, info, r, a.out)
    if totals["ticks"]:
        print(f"\n{totals['seconds'] / 60:.1f} minutes of footage, {totals['ticks']} ticks: "
              f"decode {totals['decode']:.1f}s, encode {totals['encode']:.1f}s, section {totals['section']:.2f}s. "
              f"That is {1000 * totals['decode'] / totals['ticks']:.1f} ms a tick to decode and "
              f"{1000 * totals['encode'] / totals['ticks']:.1f} ms to encode, "
              f"{totals['seconds'] / max(totals['decode'] + totals['encode'], 1e-6):.0f} times faster than playback.")


if __name__ == "__main__":
    main()
