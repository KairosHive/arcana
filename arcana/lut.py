# lut.py — a colour transfer baked into a `.cube` table
#
#   arcana-lut --content frame.png --reference photo.jpg --out grade.cube
#
# goofi's ColorLut fits Reinhard, MKL, IDT and histogram tables live from two
# images, in Rust, with nothing to download. ModFlows needs torch and a 229 MB
# checkpoint, which a realtime runtime should not carry, so it is baked here.
#
# Baking loses nothing. ModFlows moves each pixel along a flow that depends on
# the pixel's own colour and on the two images' distributions, never on its
# neighbours, so the whole transfer is a function from one colour to another.
# Running that flow on every entry of the table's lattice records it exactly,
# with no colour left for interpolation to guess: color_transfer._fit_lut3d,
# by contrast, averages the output over the colours a picture happens to hold
# and has to blur the rest in.
#
# The file is the standard `.cube` format (red fastest, 0..1 domain), so it
# also opens in Resolve, Premiere and OBS.

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image


def lattice(n: int) -> np.ndarray:
    """Every entry of an n-sided table, in `.cube` order: red fastest, then green, then blue."""
    b, g, r = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    return np.stack([r, g, b], -1).reshape(-1, 3).astype(np.float32) / (n - 1)


def bake_modflows(content, reference, n: int = 33, steps: int = 8, strength: float = 1.0) -> np.ndarray:
    """The ModFlows transfer from `content` toward `reference`, at every entry of the lattice."""
    import torch

    try:
        from . import color_transfer as _ct, modflows_net as _mn
    except ImportError:
        import color_transfer as _ct
        import modflows_net as _mn

    encoder, device = _ct._get_encoder()
    content = Image.open(content).convert("RGB") if not isinstance(content, Image.Image) else content.convert("RGB")
    reference = Image.open(reference).convert("RGB") if not isinstance(reference, Image.Image) else reference.convert("RGB")
    with torch.no_grad():
        c_params = _mn.split_params(encoder(_mn.preprocess(content, device)).flatten())
        s_params = _mn.split_params(encoder(_mn.preprocess(reference, device)).flatten())
        x = torch.from_numpy(lattice(n)).to(device)
        latent = _mn.integrate(c_params, x, steps, strength, reverse=False)
        moved = _mn.integrate(s_params, latent, steps, strength, reverse=True)
    return torch.clip(moved, 0.0, 1.0).cpu().numpy()


def write_cube(path, table: np.ndarray, n: int, title: str = "arcana") -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in table.reshape(-1, 3))
    path.write_text(f'TITLE "{title}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n{rows}\n', encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description="Bake a ModFlows colour transfer into a .cube table.")
    ap.add_argument("--content", required=True, help="A picture whose colours the table moves, such as a frame of the live source.")
    ap.add_argument("--reference", required=True, help="The picture whose colours they move toward.")
    ap.add_argument("--out", required=True, help="Where to write the .cube file.")
    ap.add_argument("--size", type=int, default=33, help="Entries along each side: 33 is the usual grade, 65 finer.")
    ap.add_argument("--steps", type=int, default=8, help="Flow steps; more is smoother and slower.")
    ap.add_argument("--strength", type=float, default=1.0, help="How far along the flow: 0 changes nothing, 1 is the full transfer.")
    args = ap.parse_args()
    table = bake_modflows(args.content, args.reference, args.size, args.steps, args.strength)
    out = write_cube(args.out, table, args.size, f"modflows {pathlib.Path(args.reference).stem}")
    print(f"wrote {out} ({args.size}^3 entries)")


if __name__ == "__main__":
    main()
