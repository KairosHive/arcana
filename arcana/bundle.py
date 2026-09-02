# bundle.py — the portable Arcana dataset bundle (.arcana)
#
# One file that holds everything about one indexed dataset, with nothing in it that
# ties it to the machine that built it.
#
# Layout (a zip; large float blocks are STORED so they can be memory-mapped):
#
#   manifest.json          format version, model identity, counts, block descriptors
#   items.jsonl            one JSON object per item, line i <-> vectors row i
#   vectors.f32            raw little-endian float32, shape (n_items, dim)
#   layout.f32             raw little-endian float32, shape (n_items, n_components)
#   thumbs/<item_id>.webp  optional previews, so a bundle is viewable without the originals
#   features/<name>.npz    optional extra feature blocks (palette, style, ...)
#
# The contract that makes it simple: items.jsonl line i corresponds to vectors.f32
# row i and layout.f32 row i. No id->row mapping to keep in sync.
#
# Items are identified by a content fingerprint, not by their path. A bundle records
# where the files were when it was built, but rebinding to a moved or renamed folder
# is a rescan, not a rebuild -- see Bundle.rebind().

from __future__ import annotations

import io
import json
import os
import zipfile
import hashlib
import mmap
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

FORMAT = "arcana.dataset"
FORMAT_VERSION = 1
SUFFIX = ".arcana"

MANIFEST_NAME = "manifest.json"
ITEMS_NAME = "items.jsonl"
VECTORS_NAME = "vectors.f32"
LAYOUT_NAME = "layout.f32"
THUMBS_PREFIX = "thumbs/"
FEATURES_PREFIX = "features/"

# Fingerprint: full content, streamed. blake2b runs at gigabytes per second, and
# ingest already reads every byte of every file to embed it, so this costs nothing
# we were not already paying -- and a sampled hash could silently collide two
# distinct photographs, which would drop one of them from the index.
_FP_CHUNK = 1 << 20  # 1 MiB
_FP_BYTES = 16


class BundleError(Exception):
    """A bundle is missing, malformed, or inconsistent with itself."""


class ModelMismatch(BundleError):
    """A query encoder does not match the encoder that built this bundle."""


# --------------------------------------------------------------------------------------
# fingerprinting
# --------------------------------------------------------------------------------------
def fingerprint(path: str) -> str:
    """Stable content id for a file, independent of its name or location."""
    h = hashlib.blake2b(digest_size=_FP_BYTES)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_FP_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Identity of the encoder that produced the vectors in a bundle."""

    id: str                       # e.g. "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    dim: int                      # embedding dimension
    modality: str = "image"       # "image" | "audio"
    normalized: bool = False      # were vectors L2-normalized before storage
    revision: str = ""            # optional pinned model revision
    preprocess: str = ""          # optional note on preprocessing variant

    def compatible_with(self, other: "ModelSpec") -> bool:
        return self.id == other.id and self.dim == other.dim and self.modality == other.modality

    @staticmethod
    def from_dict(d: dict) -> "ModelSpec":
        known = {f for f in ModelSpec.__dataclass_fields__}
        return ModelSpec(**{k: v for k, v in d.items() if k in known})


@dataclass
class Item:
    """One indexed image or audio file."""

    id: str                       # content fingerprint
    rel_path: str                 # POSIX-style, relative to the recorded root
    name: str = ""                # original basename, for display
    size: int = 0
    mtime: float = 0.0
    width: int = 0
    height: int = 0
    cluster_id: int = -1
    label: str = ""
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "Item":
        known = {f for f in Item.__dataclass_fields__}
        return Item(**{k: v for k, v in d.items() if k in known})

    @staticmethod
    def for_file(path: str, root: str, **kw) -> "Item":
        st = os.stat(path)
        return Item(
            id=fingerprint(path),
            rel_path=_rel_posix(path, root),
            name=os.path.basename(path),
            size=st.st_size,
            mtime=st.st_mtime,
            **kw,
        )


def _rel_posix(path: str, root: str) -> str:
    """Path relative to root, always with forward slashes."""
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError:
        # different drive on Windows -- fall back to the basename
        rel = os.path.basename(path)
    return rel.replace(os.sep, "/")


def _f32(a, name: str, expect_rows: int | None = None) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    if arr.ndim != 2:
        raise BundleError(f"{name} must be 2-D, got shape {arr.shape}")
    if expect_rows is not None and arr.shape[0] != expect_rows:
        raise BundleError(f"{name} has {arr.shape[0]} rows but there are {expect_rows} items")
    return arr


# --------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------
class BundleWriter:
    """
    Build a .arcana file.

        with BundleWriter("japan.arcana", name="japan", model=spec, root=IMAGES) as w:
            w.set_items(items)
            w.set_vectors(vecs)
            w.set_layout(coords, algo="tsne")
            w.add_thumbnail(item.id, webp_bytes)
            w.add_feature_block("palette", {"histogram": h, "moments": m})

    Nothing is written until close(), so a crash mid-build leaves no half-bundle.
    """

    def __init__(self, path: str, *, name: str, model: ModelSpec, root: str = "",
                 source: str = "", tool_version: str = "", created_at: str = ""):
        self.path = path
        self.name = name
        self.model = model
        self.root = root
        self.source = source
        self.tool_version = tool_version
        self.created_at = created_at

        self._items: list[Item] = []
        self._vectors: np.ndarray | None = None
        self._vector_precision: str = "f32"
        self._layout: np.ndarray | None = None
        self._layout_meta: dict = {}
        self._thumbs: dict[str, bytes] = {}
        self._features: dict[str, dict] = {}

    # -- content ------------------------------------------------------------------
    def set_items(self, items: Iterable[Item]) -> None:
        self._items = list(items)

    def set_vectors(self, vectors, *, precision: str = "f32") -> None:
        """
        Store the embedding matrix.

        `precision` records where these numbers have actually been: "f32" means
        straight from the encoder, "bf16" means they have already been through a
        bf16 round trip (a usearch index, say) and carry that quantisation
        regardless of being held as float32 here. It is provenance, not a request
        to convert -- so a reader can tell a lossless bundle from a lossy one.
        """
        self._vectors = _f32(vectors, "vectors")
        self._vector_precision = precision

    def set_layout(self, coords, *, algo: str = "", params: dict | None = None) -> None:
        self._layout = _f32(coords, "layout")
        self._layout_meta = {"algo": algo, "params": params or {},
                             "n_components": int(self._layout.shape[1])}

    def add_thumbnail(self, item_id: str, data: bytes) -> None:
        self._thumbs[item_id] = data

    def add_feature_block(self, block: str, arrays: dict) -> None:
        if "/" in block or "\\" in block:
            raise BundleError(f"feature block name must be a bare name, got {block!r}")
        self._features[block] = {k: np.asarray(v) for k, v in arrays.items()}

    # -- output -------------------------------------------------------------------
    def close(self) -> str:
        if not self._items:
            raise BundleError("refusing to write a bundle with no items")
        n = len(self._items)

        ids = [it.id for it in self._items]
        if len(set(ids)) != n:
            dupes = n - len(set(ids))
            raise BundleError(f"{dupes} duplicate item id(s); ids must be unique within a bundle")

        if self._vectors is None:
            raise BundleError("refusing to write a bundle with no vectors")
        _f32(self._vectors, "vectors", expect_rows=n)
        if self._vectors.shape[1] != self.model.dim:
            raise BundleError(
                f"vectors are {self._vectors.shape[1]}-d but model.dim is {self.model.dim}"
            )
        if self._layout is not None:
            _f32(self._layout, "layout", expect_rows=n)

        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "n_items": n,
            "model": asdict(self.model),
            "layout": self._layout_meta if self._layout is not None else None,
            "source": {"root": self.root, "note": self.source},
            "created_at": self.created_at,
            "tool_version": self.tool_version,
            "blocks": {
                "vectors": {"file": VECTORS_NAME, "dtype": "<f4",
                            "shape": list(self._vectors.shape),
                            "source_precision": self._vector_precision},
                "layout": ({"file": LAYOUT_NAME, "dtype": "<f4",
                            "shape": list(self._layout.shape)}
                           if self._layout is not None else None),
                "thumbnails": {"count": len(self._thumbs), "format": "webp"},
                "features": sorted(self._features),
            },
        }

        tmp = self.path + ".partial"
        # Store (not deflate) the float blocks: they barely compress and staying
        # uncompressed is what lets a reader memory-map them.
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
            z.writestr(ITEMS_NAME, "\n".join(json.dumps(asdict(it)) for it in self._items))
            z.writestr(zipfile.ZipInfo(VECTORS_NAME), self._vectors.tobytes(),
                       compress_type=zipfile.ZIP_STORED)
            if self._layout is not None:
                z.writestr(zipfile.ZipInfo(LAYOUT_NAME), self._layout.tobytes(),
                           compress_type=zipfile.ZIP_STORED)
            for item_id, data in self._thumbs.items():
                z.writestr(zipfile.ZipInfo(f"{THUMBS_PREFIX}{item_id}.webp"), data,
                           compress_type=zipfile.ZIP_STORED)
            for block, arrays in self._features.items():
                buf = io.BytesIO()
                np.savez_compressed(buf, **arrays)
                z.writestr(f"{FEATURES_PREFIX}{block}.npz", buf.getvalue())

        os.replace(tmp, self.path)
        return self.path

    def __enter__(self) -> "BundleWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                os.remove(self.path + ".partial")
            except OSError:
                pass


# --------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------
class Bundle:
    """
    Read a .arcana file.

        b = Bundle.open("japan.arcana")
        b.model.id            -> "laion/CLIP-ViT-H-14-..."
        b.vectors             -> (n, dim) float32, memory-mapped when possible
        b.items[0].rel_path
        b.thumbnail(item_id)  -> webp bytes or None
        b.feature("style")    -> dict of arrays or None

    Metadata is read eagerly (it is small); vectors, thumbnails and feature blocks
    are read on first access.
    """

    def __init__(self, path: str):
        self.path = path
        self._zip = zipfile.ZipFile(path, "r")
        try:
            self.manifest = json.loads(self._zip.read(MANIFEST_NAME))
        except KeyError as e:
            raise BundleError(f"{path} is not an Arcana bundle: no {MANIFEST_NAME}") from e

        if self.manifest.get("format") != FORMAT:
            raise BundleError(f"{path}: unknown format {self.manifest.get('format')!r}")
        version = int(self.manifest.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise BundleError(
                f"{path} is format version {version}; this build understands up to {FORMAT_VERSION}. "
                "Update Arcana to open it."
            )

        self.name = self.manifest.get("name", "")
        self.model = ModelSpec.from_dict(self.manifest.get("model", {}))
        self.n_items = int(self.manifest.get("n_items", 0))
        self.vector_precision = (
            ((self.manifest.get("blocks") or {}).get("vectors") or {}).get("source_precision", "f32")
        )

        raw = self._zip.read(ITEMS_NAME).decode("utf-8")
        self.items = [Item.from_dict(json.loads(ln)) for ln in raw.splitlines() if ln.strip()]
        if len(self.items) != self.n_items:
            raise BundleError(
                f"{path}: manifest says {self.n_items} items but {ITEMS_NAME} has {len(self.items)}"
            )
        self._by_id = {it.id: i for i, it in enumerate(self.items)}

        self._vectors: np.ndarray | None = None
        self._layout: np.ndarray | None = None
        self._mm: mmap.mmap | None = None
        self._fh = None

    @classmethod
    def open(cls, path: str) -> "Bundle":
        if not os.path.exists(path):
            raise BundleError(f"no bundle at {path}")
        return cls(path)

    # -- arrays -------------------------------------------------------------------
    def _data_offset(self, info: zipfile.ZipInfo) -> int | None:
        """
        Byte offset of a member's payload within the zip.

        info.header_offset points at the local file header, whose name and extra
        field lengths can differ from the central directory's -- so read the local
        header rather than trusting info.
        """
        try:
            with open(self.path, "rb") as f:
                f.seek(info.header_offset)
                hdr = f.read(30)
                if len(hdr) < 30 or hdr[:4] != b"PK\x03\x04":
                    return None
                name_len = int.from_bytes(hdr[26:28], "little")
                extra_len = int.from_bytes(hdr[28:30], "little")
            return info.header_offset + 30 + name_len + extra_len
        except OSError:
            return None

    def _read_block(self, key: str) -> np.ndarray | None:
        desc = (self.manifest.get("blocks") or {}).get(key)
        if not desc:
            return None
        shape = tuple(desc["shape"])
        member = desc["file"]
        try:
            info = self._zip.getinfo(member)
        except KeyError:
            raise BundleError(f"{self.path}: manifest lists {member} but the zip has no such member")

        count = int(np.prod(shape))
        if info.compress_type == zipfile.ZIP_STORED:
            # Map it rather than copying -- matters at 100k x 1024 floats.
            offset = self._data_offset(info)
            if offset is not None and info.file_size >= count * 4:
                if self._mm is None:
                    self._fh = open(self.path, "rb")
                    self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
                return np.frombuffer(self._mm, dtype="<f4", count=count,
                                     offset=offset).reshape(shape)

        raw = self._zip.read(member)
        if len(raw) < count * 4:
            raise BundleError(
                f"{self.path}: {member} holds {len(raw)} bytes, expected {count * 4} for shape {shape}"
            )
        return np.frombuffer(raw, dtype="<f4", count=count).reshape(shape)

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            v = self._read_block("vectors")
            if v is None:
                raise BundleError(f"{self.path}: no vectors block")
            self._vectors = v
        return self._vectors

    @property
    def layout(self) -> np.ndarray | None:
        if self._layout is None:
            self._layout = self._read_block("layout")
        return self._layout

    # -- lookups ------------------------------------------------------------------
    def index_of(self, item_id: str) -> int | None:
        return self._by_id.get(item_id)

    def thumbnail(self, item_id: str) -> bytes | None:
        try:
            return self._zip.read(f"{THUMBS_PREFIX}{item_id}.webp")
        except KeyError:
            return None

    def has_thumbnails(self) -> bool:
        return bool((self.manifest.get("blocks") or {}).get("thumbnails", {}).get("count"))

    def feature(self, block: str) -> dict | None:
        try:
            data = self._zip.read(f"{FEATURES_PREFIX}{block}.npz")
        except KeyError:
            return None
        with np.load(io.BytesIO(data)) as z:
            return {k: z[k] for k in z.files}

    def feature_blocks(self) -> list[str]:
        return list((self.manifest.get("blocks") or {}).get("features") or [])

    # -- guards -------------------------------------------------------------------
    def require_model(self, query_model: ModelSpec) -> None:
        """Refuse to mix encoders. Call this before searching with a text encoder."""
        if not self.model.compatible_with(query_model):
            raise ModelMismatch(
                f"'{self.name}' was indexed with {self.model.id} ({self.model.dim}-d, "
                f"{self.model.modality}), but the query encoder is {query_model.id} "
                f"({query_model.dim}-d, {query_model.modality}). Re-index the dataset "
                f"or load the matching encoder."
            )

    # -- relocation ---------------------------------------------------------------
    def resolve(self, root: str) -> dict[str, str]:
        """
        Map item id -> absolute path under `root`, using the recorded relative paths.
        Only ids whose file is actually present are included.
        """
        found = {}
        for it in self.items:
            p = os.path.join(root, *it.rel_path.split("/"))
            if os.path.exists(p):
                found[it.id] = p
        return found

    def rebind(self, root: str, *, extensions: Sequence[str] | None = None,
               verify: bool = True) -> dict[str, str]:
        """
        Recover item->file mapping after a folder was moved, renamed or reorganised.

        Tries the recorded relative paths first (cheap). Anything still missing is
        matched by re-fingerprinting the files under `root`, so items survive being
        moved between subfolders or renamed entirely.
        """
        found = self.resolve(root)
        missing = [it for it in self.items if it.id not in found]
        if not missing:
            return found

        wanted = {it.id for it in missing}
        by_size: dict[int, list[Item]] = {}
        for it in missing:
            by_size.setdefault(it.size, []).append(it)

        exts = {e.lower() for e in extensions} if extensions else None
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if exts and os.path.splitext(fn)[1].lower() not in exts:
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                # Size is a free pre-filter; only fingerprint plausible candidates.
                if st.st_size not in by_size:
                    continue
                try:
                    fp = fingerprint(p)
                except OSError:
                    continue
                if fp in wanted:
                    found[fp] = p
                    wanted.discard(fp)
                    if not wanted:
                        return found
        return found

    # -- integrity ----------------------------------------------------------------
    def verify(self) -> list[str]:
        """Return a list of problems; empty means the bundle is internally consistent."""
        problems: list[str] = []
        try:
            v = self.vectors
        except BundleError as e:
            problems.append(str(e))
            return problems

        if v.shape[0] != len(self.items):
            problems.append(f"vectors has {v.shape[0]} rows for {len(self.items)} items")
        if v.shape[1] != self.model.dim:
            problems.append(f"vectors are {v.shape[1]}-d but model.dim is {self.model.dim}")
        if not np.isfinite(v).all():
            problems.append("vectors contain NaN or infinity")

        lay = self.layout
        if lay is not None and lay.shape[0] != len(self.items):
            problems.append(f"layout has {lay.shape[0]} rows for {len(self.items)} items")

        for block in self.feature_blocks():
            arrays = self.feature(block)
            if arrays is None:
                problems.append(f"feature block '{block}' is listed but missing")
                continue
            ids = arrays.get("ids")
            if ids is not None and len(ids) > len(self.items):
                problems.append(f"feature block '{block}' has more rows than there are items")

        bad = self._zip.testzip()
        if bad is not None:
            problems.append(f"corrupt zip member: {bad}")
        return problems

    # -- housekeeping -------------------------------------------------------------
    def close(self) -> None:
        # Order matters: our own references to the mapped arrays go first.
        self._vectors = None
        self._layout = None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                # A caller still holds an array backed by this map. Leave it open
                # rather than corrupting their view; it is released on collection.
                pass
            self._mm = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._zip.close()

    def __enter__(self) -> "Bundle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (f"<Bundle {self.name!r} n={len(self.items)} "
                f"model={self.model.id} dim={self.model.dim} "
                f"precision={self.vector_precision}>")


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------
def list_bundles(directory: str) -> list[dict]:
    """Cheap listing for a dataset picker: manifest only, no vectors read."""
    out = []
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(SUFFIX):
            continue
        path = os.path.join(directory, fn)
        try:
            with zipfile.ZipFile(path, "r") as z:
                m = json.loads(z.read(MANIFEST_NAME))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            continue
        if m.get("format") != FORMAT:
            continue
        model = m.get("model") or {}
        out.append({
            "path": path,
            "name": m.get("name", os.path.splitext(fn)[0]),
            "n_items": m.get("n_items", 0),
            "modality": model.get("modality", "image"),
            "model_id": model.get("id", ""),
            "dim": model.get("dim", 0),
            "n_components": (m.get("layout") or {}).get("n_components", 0),
            "format_version": m.get("format_version", 0),
            "vector_precision": ((m.get("blocks") or {}).get("vectors") or {}).get("source_precision", "f32"),
        })
    return out
