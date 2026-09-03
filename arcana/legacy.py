# legacy.py — convert pre-bundle datasets into portable .arcana bundles
#
# The old layout spread one dataset across several machine-coupled files:
#
#   databases/index_<name>_<mod>.pkl          pickle of (usearch blob, {int: ABSOLUTE path})
#   latents/latent_space_<name>_<mod>_<n>d.pkl  pandas pickle, ABSOLUTE paths again
#   databases/features_<name>_palette.npz     rows keyed by the index's integer ids
#   databases/features_<name>_style.npz       same
#
# Nothing recorded which model produced the vectors, and every path was absolute,
# so a dataset could not survive being moved. This module reads all of that and
# writes a single self-describing bundle.
#
# Usage:
#   python -m arcana.legacy --list
#   python -m arcana.legacy --all --thumbnails
#   python -m arcana.legacy --name japan --modality image

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass

import numpy as np

from .bundle import BundleWriter, Item, ModelSpec, SUFFIX, fingerprint

try:
    from . import paths as _paths
except ImportError:                      # running as a loose script
    import paths as _paths

# Resolved through paths.py, not from __file__. Deriving these from the module
# location means a packaged app looks inside its own read-only install
# directory: the frozen build reported "no dataset named 'japan'" for a dataset
# that was loaded and on screen, because discover() was searching
# _internal/arcana/databases while everything else honoured ARCANA_DATA_DIR.
APP_ROOT = _paths.APP_ROOT
DB_DIR = _paths.subdir("databases")
LATENTS_DIR = _paths.subdir("latents")
BUNDLES_DIR = _paths.subdir("bundles")

# What db.py hardcoded at the time these datasets were built. Recording it is the
# whole point: without it a bundle cannot know which text encoder can query it.
LEGACY_MODELS = {
    "image": ModelSpec(id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", dim=1024,
                       modality="image", normalized=False),
    "audio": ModelSpec(id="laion/clap-htsat-fused", dim=512,
                       modality="audio", normalized=False),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

_INDEX_RE = re.compile(r"^index_(.+)_(image|audio)\.pkl$")
_LATENT_RE = re.compile(r"^latent_space_(.+)_(image|audio)_(\d+)d\.pkl$")


@dataclass
class LegacyDataset:
    name: str
    modality: str
    index_path: str
    latent_paths: dict[int, str]      # n_components -> path
    palette_path: str | None
    style_path: str | None

    @property
    def key(self) -> str:
        return f"{self.name}_{self.modality}"


def discover(db_dir: str = DB_DIR, latents_dir: str = LATENTS_DIR) -> list[LegacyDataset]:
    """Find every legacy dataset that has at least an index."""
    indexes: dict[tuple[str, str], str] = {}
    if os.path.isdir(db_dir):
        for fn in os.listdir(db_dir):
            m = _INDEX_RE.match(fn)
            if m:
                indexes[(m.group(1), m.group(2))] = os.path.join(db_dir, fn)

    latents: dict[tuple[str, str], dict[int, str]] = {}
    if os.path.isdir(latents_dir):
        for fn in os.listdir(latents_dir):
            m = _LATENT_RE.match(fn)
            if m:
                latents.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = \
                    os.path.join(latents_dir, fn)

    out = []
    for (name, modality), index_path in sorted(indexes.items()):
        pal = os.path.join(db_dir, f"features_{name}_palette.npz")
        sty = os.path.join(db_dir, f"features_{name}_style.npz")
        out.append(LegacyDataset(
            name=name,
            modality=modality,
            index_path=index_path,
            latent_paths=latents.get((name, modality), {}),
            palette_path=pal if os.path.exists(pal) else None,
            style_path=sty if os.path.exists(sty) else None,
        ))
    return out


def _norm(p: str) -> str:
    """Comparison key for a path recorded on this machine."""
    return os.path.normcase(os.path.normpath(p))


def _common_root(paths: list[str]) -> str:
    """Deepest directory containing every path, so rel_paths stay meaningful."""
    existing = [p for p in paths if p]
    if not existing:
        return ""
    try:
        root = os.path.commonpath([os.path.abspath(p) for p in existing])
    except ValueError:
        return ""                       # mixed drives on Windows
    return root if os.path.isdir(root) else os.path.dirname(root)


def _thumbnail(path: str, max_side: int = 192, quality: int = 80) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            import io
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=quality, method=4)
            return buf.getvalue()
    except Exception:
        return None


def convert(ds: LegacyDataset, out_dir: str = BUNDLES_DIR, *,
            thumbnails: bool = False, n_components: int | None = None,
            verbose: bool = True) -> dict:
    """
    Convert one legacy dataset to a bundle. Returns a report dict.

    Items whose source file is missing are still carried, with their vector and
    coordinates intact -- the bundle stays complete and a later rebind() can
    reattach the pixels. They are counted in the report so nothing is silent.
    """
    from usearch.index import Index

    report = {"name": ds.name, "modality": ds.modality, "out": None,
              "n_items": 0, "missing_files": 0, "unmatched_coords": 0,
              "features": [], "warnings": []}

    with open(ds.index_path, "rb") as f:
        blob, idx2path = pickle.load(f)
    index = Index.restore(blob)

    # Legacy key type drifted between writers; normalise to int and keep the order stable.
    keys = sorted(int(k) for k in idx2path.keys())
    raw_paths = {int(k): str(v) for k, v in idx2path.items()}

    vecs = np.asarray([index.get(k) for k in keys], dtype=np.float32)
    if vecs.ndim == 3 and vecs.shape[1] == 1:       # usearch may return (1, dim) per key
        vecs = vecs[:, 0, :]
    if vecs.ndim != 2:
        raise SystemExit(f"{ds.key}: could not read vectors from the index (got shape {vecs.shape})")

    model = LEGACY_MODELS.get(ds.modality)
    if model is None:
        raise SystemExit(f"{ds.key}: unknown modality {ds.modality!r}")
    if vecs.shape[1] != model.dim:
        report["warnings"].append(
            f"index is {vecs.shape[1]}-d but {model.id} is {model.dim}-d; "
            f"recording the actual dimension"
        )
        model = ModelSpec(id=model.id, dim=int(vecs.shape[1]), modality=model.modality,
                          normalized=model.normalized)

    # --- coordinates, labels, clusters from the latent DataFrame ---------------
    coords = None
    meta_by_path: dict[str, dict] = {}
    if ds.latent_paths:
        want = n_components if n_components in ds.latent_paths else min(ds.latent_paths)
        import pandas as pd
        df = pd.read_pickle(ds.latent_paths[want])
        comp_cols = [c for c in ("x", "y", "z") if c in df.columns][:want]
        for _, row in df.iterrows():
            meta_by_path[_norm(str(row["path"]))] = {
                "coords": [float(row[c]) for c in comp_cols],
                "cluster_id": int(row["cluster_id"]) if "cluster_id" in df.columns else -1,
                "label": str(row["label"]) if "label" in df.columns else "",
            }
        report["layout_dims"] = len(comp_cols)
    else:
        report["warnings"].append("no latent_space file found; bundle will have no 2-D layout")

    # --- build items ----------------------------------------------------------
    root = _common_root([raw_paths[k] for k in keys])
    exts = IMAGE_EXTS if ds.modality == "image" else AUDIO_EXTS
    items: list[Item] = []
    coord_rows: list[list[float]] = []
    n_dims = report.get("layout_dims", 0)
    seen_ids: dict[str, int] = {}
    drop_rows: list[int] = []

    for row, k in enumerate(keys):
        src = raw_paths[k]
        nk = _norm(src)
        meta = meta_by_path.get(nk)
        if meta is None:
            report["unmatched_coords"] += 1

        if os.path.exists(src):
            try:
                item = Item.for_file(src, root)
            except OSError:
                item = None
        else:
            item = None

        if item is None:
            report["missing_files"] += 1
            # No file to fingerprint: fall back to a stable id derived from the
            # recorded path, so the row still has a unique key.
            import hashlib
            item = Item(
                id="p" + hashlib.blake2b(nk.encode(), digest_size=15).hexdigest(),
                rel_path=os.path.relpath(src, root).replace(os.sep, "/") if root else os.path.basename(src),
                name=os.path.basename(src),
            )

        if os.path.splitext(item.name)[1].lower() not in exts and item.name:
            report["warnings"].append(f"unexpected extension for {ds.modality}: {item.name}")

        if item.id in seen_ids:
            # Byte-identical duplicates existed in the source folder. Keep the first;
            # a bundle's ids must be unique.
            drop_rows.append(row)
            continue
        seen_ids[item.id] = row

        item.cluster_id = meta["cluster_id"] if meta else -1
        item.label = meta["label"] if meta else ""
        items.append(item)
        if n_dims:
            coord_rows.append((meta["coords"] if meta else [0.0] * n_dims)[:n_dims])

    if drop_rows:
        report["warnings"].append(
            f"{len(drop_rows)} duplicate file(s) with identical content collapsed to one entry"
        )
        keep = [r for r in range(len(keys)) if r not in set(drop_rows)]
        vecs = vecs[keep]
        old_row_of_new = keep
    else:
        old_row_of_new = list(range(len(keys)))

    if n_dims:
        coords = np.asarray(coord_rows, dtype=np.float32)

    report["n_items"] = len(items)

    # --- feature blocks, re-keyed from legacy ids to bundle row order ----------
    #     legacy `ids` are usearch keys; a bundle's rows are positional.
    key_to_new_row = {keys[old]: new for new, old in enumerate(old_row_of_new)}

    def remap(npz_path: str, block: str) -> dict | None:
        try:
            z = np.load(npz_path)
        except Exception as e:
            report["warnings"].append(f"could not read {os.path.basename(npz_path)}: {e}")
            return None
        with z:
            if "ids" not in z.files:
                report["warnings"].append(f"{block} features have no ids array; skipping")
                return None
            old_ids = z["ids"].astype(int)
            new_rows, keep_pos = [], []
            for pos, oid in enumerate(old_ids):
                nr = key_to_new_row.get(int(oid))
                if nr is not None:
                    new_rows.append(nr)
                    keep_pos.append(pos)
            if not keep_pos:
                report["warnings"].append(f"{block} features matched no items; skipping")
                return None
            if len(keep_pos) != len(old_ids):
                report["warnings"].append(
                    f"{block}: {len(old_ids) - len(keep_pos)} feature row(s) had no matching item"
                )
            out = {"ids": np.asarray(new_rows, dtype=np.int32)}
            keep_arr = np.asarray(keep_pos, dtype=int)
            for name in z.files:
                if name == "ids":
                    continue
                arr = z[name]
                # Per-item arrays get filtered; shared arrays (a PCA basis) are copied whole.
                out[name] = arr[keep_arr] if arr.shape[:1] == old_ids.shape else arr
            return out

    features: dict[str, dict] = {}
    if ds.palette_path:
        blk = remap(ds.palette_path, "palette")
        if blk:
            features["palette"] = blk
            report["features"].append("palette")
    if ds.style_path:
        blk = remap(ds.style_path, "style")
        if blk:
            features["style"] = blk
            report["features"].append("style")

    # --- write ----------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ds.name}_{ds.modality}{SUFFIX}")
    with BundleWriter(out_path, name=ds.name, model=model, root=root,
                      source=f"migrated from {os.path.basename(ds.index_path)}",
                      tool_version="arcana.legacy/1") as w:
        w.set_items(items)
        # db.py built these indexes with Index(ndim=..., metric="cos") and no dtype,
        # and usearch defaults to bf16 -- so these vectors already carry ~1e-2
        # quantisation error and the original f32 values are not recoverable.
        # Record that rather than let a migrated bundle look lossless.
        w.set_vectors(vecs, precision="bf16")
        if coords is not None:
            w.set_layout(coords, algo="tsne", params={"migrated": True})
        for block, arrays in features.items():
            w.add_feature_block(block, arrays)
        if thumbnails and ds.modality == "image":
            made = 0
            for it in items:
                src = os.path.join(root, *it.rel_path.split("/")) if root else it.rel_path
                data = _thumbnail(src)
                if data:
                    w.add_thumbnail(it.id, data)
                    made += 1
                if verbose and made and made % 500 == 0:
                    print(f"      thumbnails: {made}/{len(items)}", flush=True)
            report["thumbnails"] = made

    report["out"] = out_path
    report["size_mb"] = round(os.path.getsize(out_path) / 1e6, 1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert legacy Arcana datasets to portable bundles.")
    ap.add_argument("--list", action="store_true", help="show what would be converted and exit")
    ap.add_argument("--all", action="store_true", help="convert every discovered dataset")
    ap.add_argument("--name", help="convert only this dataset name")
    ap.add_argument("--modality", choices=["image", "audio"], help="restrict to one modality")
    ap.add_argument("--out", default=BUNDLES_DIR, help="output directory for bundles")
    ap.add_argument("--thumbnails", action="store_true",
                    help="embed 192px previews so the bundle is viewable without the originals")
    ap.add_argument("--components", type=int, default=None, help="prefer this layout dimensionality")
    args = ap.parse_args(argv)

    found = discover()
    if args.name:
        found = [d for d in found if d.name == args.name]
    if args.modality:
        found = [d for d in found if d.modality == args.modality]

    if not found:
        print("No legacy datasets found.")
        return 1

    if args.list or not (args.all or args.name):
        print(f"{len(found)} legacy dataset(s):\n")
        for d in found:
            dims = ",".join(f"{k}D" for k in sorted(d.latent_paths)) or "no layout"
            extra = "+".join(x for x, p in (("palette", d.palette_path), ("style", d.style_path)) if p)
            print(f"  {d.name:20s} {d.modality:6s} {dims:12s} {extra}")
        if args.list:
            return 0
        print("\nRe-run with --all, or --name <name>, to convert.")
        return 0

    failures = 0
    for d in found:
        print(f"\n=== {d.name} ({d.modality}) ===", flush=True)
        try:
            r = convert(d, args.out, thumbnails=args.thumbnails, n_components=args.components)
        except Exception as e:
            failures += 1
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        print(f"  -> {r['out']}  ({r['size_mb']} MB, {r['n_items']} items)")
        if r["features"]:
            print(f"     features: {', '.join(r['features'])}")
        if r.get("thumbnails"):
            print(f"     thumbnails: {r['thumbnails']}")
        if r["missing_files"]:
            print(f"     NOTE: {r['missing_files']} source file(s) not on disk; "
                  f"vectors kept, pixels recoverable via rebind()")
        if r["unmatched_coords"]:
            print(f"     NOTE: {r['unmatched_coords']} item(s) had no matching layout row")
        for wmsg in r["warnings"]:
            print(f"     WARN: {wmsg}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
