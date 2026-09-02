# relocate.py — re-point a dataset after its media folder moved
#
# Datasets built before the portable bundle format store absolute paths inside
# the index pickle and the latent DataFrame, so moving, renaming or copying the
# photo folder to another machine breaks them completely. Bundles store a
# content fingerprint plus a path relative to a recorded root, so they only need
# to be told where the root went.
#
# This tool handles both, and never rebuilds anything: no re-encoding, no GPU.
#
#   arcana-relocate --list
#   arcana-relocate --name japan --modality image --new-root D:\Photos\japan
#   arcana-relocate --bundle arcana/bundles/japan_image.arcana --new-root D:\Photos --dry-run
#
# Matching happens in three passes, cheapest first:
#   1. same relative layout under the new root
#   2. same filename anywhere under it
#   3. same content, by fingerprint (only for whatever is still missing)

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from collections import defaultdict

import numpy as np

try:
    from . import paths as _paths
    from .bundle import Bundle, BundleWriter, Item, ModelSpec, SUFFIX, fingerprint
    from .legacy import discover, _norm, _common_root
except ImportError:  # running as a loose script
    import paths as _paths
    from bundle import Bundle, BundleWriter, Item, ModelSpec, SUFFIX, fingerprint
    from legacy import discover, _norm, _common_root


def dataset_health(name: str, modality: str = "image", sample: int = 200,
                   db_dir: str | None = None, latents_dir: str | None = None) -> dict:
    """
    Is this dataset's media still where the index says it is?

    Samples rather than stat-ing all 82k paths, because this runs every time the
    dataset dropdown changes. `sample=0` checks everything.

    Returns {ok, total, checked, present, missing, root, error}.
    """
    out = {"ok": True, "total": 0, "checked": 0, "present": 0, "missing": 0,
           "root": "", "error": ""}
    kw = {}
    if db_dir is not None:
        kw["db_dir"] = db_dir
    if latents_dir is not None:
        kw["latents_dir"] = latents_dir
    match = [d for d in discover(**kw) if d.name == name and d.modality == modality]
    if not match:
        out["error"] = f"no dataset named {name!r} ({modality})"
        out["ok"] = False
        return out
    try:
        with open(match[0].index_path, "rb") as f:
            _blob, idx2path = pickle.load(f)
    except Exception as e:
        out["error"] = f"could not read the index: {e}"
        out["ok"] = False
        return out

    all_paths = [str(v) for v in idx2path.values()]
    out["total"] = len(all_paths)
    out["root"] = _common_root(all_paths)

    if sample and len(all_paths) > sample:
        step = max(1, len(all_paths) // sample)
        probe = all_paths[::step][:sample]
    else:
        probe = all_paths
    out["checked"] = len(probe)
    out["present"] = sum(1 for p in probe if os.path.exists(p))
    out["missing"] = out["checked"] - out["present"]
    out["ok"] = out["missing"] == 0
    return out


def _index_new_root(new_root: str) -> tuple[dict, dict]:
    """Index the destination once: basename -> [paths], size -> [paths]."""
    by_name: dict[str, list[str]] = defaultdict(list)
    by_size: dict[int, list[str]] = defaultdict(list)
    for dirpath, _dirs, files in os.walk(new_root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            by_name[fn.lower()].append(p)
            try:
                by_size[os.path.getsize(p)].append(p)
            except OSError:
                pass
    return by_name, by_size


def _resolve_paths(old_paths: list[str], new_root: str, verify_hashes: bool = True) -> tuple[dict, list]:
    """
    Map each old absolute path to a file under new_root.

    Returns (old -> new, still_missing).
    """
    new_root = os.path.abspath(new_root)
    old_root = _common_root(old_paths)
    mapping: dict[str, str] = {}
    missing: list[str] = []

    # Pass 1 -- the folder simply moved, structure intact.
    for p in old_paths:
        rel = os.path.relpath(p, old_root) if old_root else os.path.basename(p)
        cand = os.path.join(new_root, rel)
        if os.path.isfile(cand):
            mapping[p] = cand
        else:
            missing.append(p)

    if not missing:
        return mapping, []

    by_name, by_size = _index_new_root(new_root)
    taken = set(mapping.values())

    # Pass 2 -- same filename somewhere under the new root.
    still: list[str] = []
    for p in missing:
        cands = [c for c in by_name.get(os.path.basename(p).lower(), []) if c not in taken]
        if len(cands) == 1:
            mapping[p] = cands[0]
            taken.add(cands[0])
        elif len(cands) > 1:
            # Ambiguous: prefer one whose parent folder name also matches.
            parent = os.path.basename(os.path.dirname(p)).lower()
            best = [c for c in cands if os.path.basename(os.path.dirname(c)).lower() == parent]
            if len(best) == 1:
                mapping[p] = best[0]
                taken.add(best[0])
            else:
                still.append(p)
        else:
            still.append(p)

    # Pass 3 -- content match, for files that were renamed.
    if still and verify_hashes:
        want: dict[str, str] = {}
        for p in still:
            try:
                want[fingerprint(p)] = p       # only works if the OLD file is still readable
            except OSError:
                pass
        if want:
            for size, cands in by_size.items():
                for c in cands:
                    if c in taken:
                        continue
                    try:
                        fp = fingerprint(c)
                    except OSError:
                        continue
                    src = want.pop(fp, None)
                    if src is not None:
                        mapping[src] = c
                        taken.add(c)
                    if not want:
                        break
                if not want:
                    break
        still = [p for p in still if p not in mapping]

    return mapping, still


# --------------------------------------------------------------------------------------
# bundles
# --------------------------------------------------------------------------------------
def relocate_bundle(bundle_path: str, new_root: str, dry_run: bool = False) -> dict:
    """Point a bundle at a new media root, matching items by content."""
    report = {"target": bundle_path, "kind": "bundle", "total": 0, "found": 0, "missing": 0,
              "changed": False}
    with Bundle.open(bundle_path) as b:
        report["total"] = len(b)
        found = b.rebind(new_root)
        report["found"] = len(found)
        report["missing"] = len(b) - len(found)
        if dry_run or not found:
            return report

        root = os.path.abspath(new_root)
        items = []
        for it in b.items:
            p = found.get(it.id)
            new_it = Item(id=it.id,
                          rel_path=(os.path.relpath(p, root).replace(os.sep, "/") if p else it.rel_path),
                          name=(os.path.basename(p) if p else it.name),
                          size=it.size, mtime=it.mtime, width=it.width, height=it.height,
                          cluster_id=it.cluster_id, label=it.label, extra=it.extra)
            items.append(new_it)

        vectors = np.array(b.vectors)
        layout = None if b.layout is None else np.array(b.layout)
        lay_meta = b.manifest.get("layout") or {}
        features = {blk: b.feature(blk) for blk in b.feature_blocks()}
        thumbs = {it.id: b.thumbnail(it.id) for it in b.items} if b.has_thumbnails() else {}
        model = b.model
        name = b.name
        precision = b.vector_precision

    tmp = bundle_path + ".relocating"
    with BundleWriter(tmp, name=name, model=model, root=root,
                      source=f"relocated to {root}", tool_version="arcana.relocate/1") as w:
        w.set_items(items)
        w.set_vectors(vectors, precision=precision)
        if layout is not None:
            w.set_layout(layout, algo=lay_meta.get("algo", ""), params=lay_meta.get("params") or {})
        for blk, arrays in features.items():
            if arrays:
                w.add_feature_block(blk, arrays)
        for item_id, data in thumbs.items():
            if data:
                w.add_thumbnail(item_id, data)

    shutil.move(tmp, bundle_path)
    report["changed"] = True
    return report


# --------------------------------------------------------------------------------------
# legacy pickles
# --------------------------------------------------------------------------------------
def relocate_legacy(ds, new_root: str, dry_run: bool = False, backup: bool = True) -> dict:
    """Rewrite the absolute paths inside a legacy index + latent pickle."""
    import pandas as pd

    report = {"target": ds.key, "kind": "legacy", "total": 0, "found": 0, "missing": 0,
              "changed": False, "files": []}

    with open(ds.index_path, "rb") as f:
        blob, idx2path = pickle.load(f)
    old_paths = [str(v) for v in idx2path.values()]
    report["total"] = len(old_paths)

    mapping, missing = _resolve_paths(old_paths, new_root)
    report["found"] = len(mapping)
    report["missing"] = len(missing)
    if dry_run or not mapping:
        return report

    new_idx2path = {int(k): mapping.get(str(v), str(v)) for k, v in idx2path.items()}
    if backup and not os.path.exists(ds.index_path + ".bak"):
        shutil.copy2(ds.index_path, ds.index_path + ".bak")
    with open(ds.index_path, "wb") as f:
        pickle.dump((blob, new_idx2path), f)
    report["files"].append(ds.index_path)

    norm_map = {_norm(k): v for k, v in mapping.items()}
    for _dim, lat_path in sorted(ds.latent_paths.items()):
        df = pd.read_pickle(lat_path)
        if "path" not in df.columns:
            continue
        df["path"] = [norm_map.get(_norm(str(p)), str(p)) for p in df["path"]]
        if backup and not os.path.exists(lat_path + ".bak"):
            shutil.copy2(lat_path, lat_path + ".bak")
        df.to_pickle(lat_path)
        report["files"].append(lat_path)

    report["changed"] = True
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _bundle_entries() -> list[dict]:
    try:
        from .bundle import list_bundles
    except ImportError:
        from bundle import list_bundles
    return list_bundles(_paths.subdir("bundles"))


def _describe_state(paths_sample: list[str]) -> str:
    if not paths_sample:
        return "no items"
    present = sum(1 for p in paths_sample if os.path.exists(p))
    return f"{present}/{len(paths_sample)} sampled files present"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Re-point a dataset after its media folder moved. Never re-encodes.")
    ap.add_argument("--list", action="store_true", help="show every dataset and whether its files are reachable")
    ap.add_argument("--name", help="legacy dataset name to relocate")
    ap.add_argument("--modality", choices=["image", "audio"], default="image")
    ap.add_argument("--bundle", help="path to a .arcana bundle to relocate")
    ap.add_argument("--all-bundles", action="store_true", help="relocate every bundle")
    ap.add_argument("--new-root", help="the folder the media now lives in")
    ap.add_argument("--dry-run", action="store_true", help="report what would match, change nothing")
    ap.add_argument("--no-backup", action="store_true", help="skip writing .bak copies of legacy pickles")
    args = ap.parse_args(argv)

    if args.list:
        print("Legacy datasets (absolute paths baked in):")
        for d in discover():
            try:
                with open(d.index_path, "rb") as f:
                    _blob, idx2path = pickle.load(f)
                sample = [str(v) for v in list(idx2path.values())[:25]]
                root = _common_root([str(v) for v in idx2path.values()])
            except Exception as e:
                print(f"  {d.key:26s} unreadable ({e})")
                continue
            print(f"  {d.key:26s} {_describe_state(sample):32s} root: {root}")

        entries = _bundle_entries()
        print(f"\nPortable bundles ({len(entries)}):")
        for e in entries:
            try:
                with Bundle.open(e["path"]) as b:
                    root = b.manifest.get("source", {}).get("root", "")
                    sample = [os.path.join(root, *it.rel_path.split("/")) for it in b.items[:25]]
                print(f"  {e['name']:26s} {_describe_state(sample):32s} root: {root}")
            except Exception as ex:
                print(f"  {e['name']:26s} unreadable ({ex})")
        return 0

    if not args.new_root:
        ap.error("--new-root is required (or use --list)")
    if not os.path.isdir(args.new_root):
        ap.error(f"--new-root is not a directory: {args.new_root}")

    reports = []
    if args.bundle:
        reports.append(relocate_bundle(args.bundle, args.new_root, args.dry_run))
    elif args.all_bundles:
        for e in _bundle_entries():
            reports.append(relocate_bundle(e["path"], args.new_root, args.dry_run))
    elif args.name:
        matches = [d for d in discover() if d.name == args.name and d.modality == args.modality]
        if not matches:
            print(f"No legacy dataset named {args.name!r} ({args.modality}).")
            return 1
        reports.append(relocate_legacy(matches[0], args.new_root, args.dry_run,
                                       backup=not args.no_backup))
    else:
        ap.error("give --name, --bundle, or --all-bundles")

    failed = 0
    for r in reports:
        verb = "would match" if args.dry_run else ("relocated" if r["changed"] else "no change")
        print(f"\n{r['target']}  [{r['kind']}]")
        print(f"  {verb}: {r['found']}/{r['total']} items")
        if r["missing"]:
            print(f"  still missing: {r['missing']} (not found under {args.new_root})")
            if r["found"] == 0:
                failed += 1
        for fp in r.get("files", []):
            print(f"  rewrote {fp}" + ("  (backup at .bak)" if not args.no_backup else ""))
    if args.dry_run:
        print("\nDry run: nothing was written.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
