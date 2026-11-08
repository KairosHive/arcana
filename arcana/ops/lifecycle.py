# lifecycle.py -- making, growing and reworking datasets
#
# These were the Datasets tab's job bodies and four console scripts
# (arcana-build-latent, arcana-extend, arcana-finish, arcana-marks), plus
# arcana-relocate and arcana-migrate. Each is an effect: it writes a dataset's
# index, layout and features, which are files no transaction snapshots, since
# a dataset is rebuilt as a whole rather than edited. They run as jobs by
# default, because indexing takes minutes to hours. `job wait` returns the
# result and `job cancel` stops one at its next safe point.

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field

from . import archive
from .archive import dataset_arg
from .effects import run_effect
from .registry import Args, Context, Result, op, positional

try:
    from .. import core
    from .. import db as _db
    from .. import models as _models
except ImportError:                                   # pragma: no cover
    import core
    import db as _db
    import models as _models

Modality = Literal["image", "audio"]
Labels = Literal["builtin", "folders", "none"]
Feature = Literal["palette", "style"]


def _db_progress(progress):
    """db.py reports (fraction, message, done, total). The ops report (fraction, message)."""
    if progress is None:
        return None

    def cb(f, m=None, d=0, t=0):
        progress(f, (m or None) if not t else f"{m or 'Working'} ({d:,} of {t:,})")
    return cb


def _cancelled(progress):
    return getattr(progress, "cancelled", None) or (lambda: False)


def _dirs() -> dict:
    """The data folders as they are now. legacy.py fixes its defaults at import."""
    try:
        from .. import paths as _p
    except ImportError:                               # pragma: no cover
        import paths as _p
    return {"db_dir": _p.subdir("databases"), "latents_dir": _p.subdir("latents")}


def _paths_bundles() -> str:
    try:
        from .. import paths as _p
    except ImportError:                               # pragma: no cover
        import paths as _p
    return _p.ensure_dir(_p.subdir("bundles"))


def _discover():
    try:
        from .. import legacy as _legacy
    except ImportError:                               # pragma: no cover
        import legacy as _legacy
    return _legacy.discover(**_dirs())


def _features(feats) -> str:
    return ",".join(["clip"] + sorted(set(feats or []) & {"palette", "style"}))


def _labels(choice: str, folder: str | None):
    """None means the built-in word list, "" leaves clusters numbered."""
    if choice == "none":
        return ""
    if choice == "folders" and folder:
        words = _db.folder_vocabulary(folder)
        return ",".join(words) if words else None
    return None


def _check_free(name: str) -> None:
    if not name or not all(c.isalnum() or c in "-_ " for c in name):
        raise ValueError("A dataset name holds letters, numbers, spaces, - and _ only.")


class DatasetJobResult(Result):
    dataset: str
    summary: str
    details: dict = Field(default_factory=dict)

    def text(self) -> str:
        return f"{self.dataset}: {self.summary}\n"


def _done(ds_id: str, summary: str, details: dict | None = None):
    safe = {k: v for k, v in (details or {}).items()
            if isinstance(v, (str, int, float, bool)) or v is None}
    return lambda tx, _payload=None: DatasetJobResult(dataset=ds_id, summary=summary, details=safe)


# ─────────────────────────── create ───────────────────────────

class CreateArgs(Args):
    name: str = positional(0, description="The new dataset's name.")
    folder: str = Field(description="The folder of pictures (or sounds) to index.",
                        json_schema_extra={"metavar": "PATH"})
    modality: Modality = Field("image", description="image or audio.")
    model: str | None = Field(None, description="The encoder, from `model list`. Defaults to "
                                                "the right one for this machine.")
    feature: list[Feature] = Field(default_factory=list,
                                   description="Extra features to build: palette, style. "
                                               "Repeat for both.")
    thumbnails: bool = Field(False, description="Cache thumbnails for a faster map.")
    k: int = Field(0, ge=0, le=200, description="Clusters. 0 chooses a number for you.")
    labels: Labels = Field("builtin", description="How clusters are named: builtin words, "
                                                  "the folders' names, or none.")
    dims: Literal[2, 3] = Field(2, description="The map's dimensions.")
    background: bool = Field(True, description="Run as a job. --no-background waits.")


@op("dataset create", "effect", CreateArgs, Result,
    doc="""
    Index a folder into a new dataset. It encodes every picture with the
    encoder, lays out the map, names the clusters and reads ON1 and Lightroom
    marks, plus palette and style features when asked. It runs as a job. The
    files are read and never moved.
    """,
    examples=('dataset create holiday --folder "D:/Photos/2024 holiday" --feature palette',))
def _create(a: CreateArgs, ctx: Context) -> Result:
    _check_free(a.name)
    folder = os.path.abspath(a.folder)
    if not os.path.isdir(folder):
        raise ValueError(f"Not a folder: {a.folder}")
    if any(d["name"] == a.name and d["modality"] == a.modality for d in core.find_datasets()):
        raise ValueError(f"A dataset called {a.name} exists. Use `dataset extend {a.name}` to "
                         f"add new files to it, or pick another name.")
    model_id = a.model or _models.default_for(a.modality).id
    if a.model and _models.get(a.model) is None:
        raise ValueError(f"No encoder {a.model!r}. `model list` shows them.")
    labels = _labels(a.labels, folder)

    def compute(progress):
        if progress:
            progress(0.0, "Preparing the encoder")
        _models.ensure_model(model_id, _Handle(progress))
        if a.labels != "none":
            _db.warm_label_cache(model_id, a.modality)
        return _db.index_dataset(folder, a.name, modality=a.modality, n_components=a.dims,
                                 model_id=model_id, features=_features(a.feature),
                                 thumbnails=a.thumbnails, k=a.k, labels=labels,
                                 progress=_db_progress(progress),
                                 should_cancel=_cancelled(progress)) or {}

    def record(tx, result):
        ds_id = archive.dataset_id(a.name, a.modality)
        n = result.get("n_items") or result.get("count") or ""
        return _done(ds_id, f"indexed{f' {n:,} items' if isinstance(n, int) else ''} from {folder}.",
                     result)(tx)

    return run_effect(ctx, f"dataset create {a.name}", compute, record, a.background, kind="index")


class _Handle:
    """What models.ensure_model expects of a job handle, over an ops progress callback."""

    def __init__(self, progress):
        self.progress = progress

    def update(self, fraction=None, message=None, **_):
        if self.progress:
            self.progress(fraction, message)

    def raise_if_cancelled(self):
        if _cancelled(self.progress)():
            from .. import jobs
            raise jobs.Cancelled()

    @property
    def cancelled(self):
        return _cancelled(self.progress)()


# ─────────────────────────── extend, finish, marks ───────────────────────────

class ExtendArgs(Args):
    dataset: str = dataset_arg()
    folder: str | None = Field(None, description="Where to look for new files. Defaults to the "
                                                 "folder the dataset already comes from.",
                               json_schema_extra={"metavar": "PATH"})
    prune: bool = Field(False, description="Also drop pictures whose files are gone.")
    background: bool = Field(True, description="Run as a job. --no-background waits.")


@op("dataset extend", "effect", ExtendArgs, Result,
    doc="""
    Add the files that appeared in a dataset's folder since it was indexed.
    Only the new files are encoded, then the map is laid out again over the
    whole collection.
    """,
    examples=("dataset extend japan --no-background",))
def _extend(a: ExtendArgs, ctx: Context) -> Result:
    try:
        from ..relocate import dataset_health
    except ImportError:                               # pragma: no cover
        from relocate import dataset_health
    name, modality = archive.split_dataset_id(a.dataset)
    archive.open_dataset(a.dataset)
    root = a.folder or (dataset_health(name, modality, sample=1, **_dirs()) or {}).get("root") or ""
    if not root or not os.path.isdir(root):
        raise ValueError(f"Cannot find {root or 'the folder'} for {a.dataset}. If the files "
                         f"moved, `dataset relocate {a.dataset} --root <folder>` first.")

    def compute(progress):
        return _db.extend_dataset(root, name, modality=modality, prune_missing=a.prune,
                                  progress=_db_progress(progress),
                                  should_cancel=_cancelled(progress)) or {}

    def record(tx, result):
        added = result.get("added", 0)
        return _done(a.dataset, f"added {added:,} new file(s) from {root}." if added
                     else f"nothing new in {root}.", result)(tx)

    return run_effect(ctx, f"dataset extend {a.dataset}", compute, record, a.background, kind="index")


class FinishArgs(Args):
    dataset: str = dataset_arg()
    k: int = Field(0, ge=0, le=200, description="Clusters. 0 chooses a number for you.")
    labels: Labels = Field("builtin", description="How clusters are named: builtin words, "
                                                  "the folders' names, or none.")
    dims: Literal[2, 3] = Field(2, description="The map's dimensions.")
    feature: list[Feature] = Field(default_factory=list,
                                   description="Features to (re)build: palette, style.")
    thumbnails: bool = Field(False, description="Cache thumbnails for a faster map.")
    background: bool = Field(True, description="Run as a job. --no-background waits.")


@op("dataset finish", "effect", FinishArgs, Result,
    doc="""
    Lay out a dataset again from the index it has, without re-encoding. It
    finishes an indexing run that was cut short, or reworks the clusters,
    their names, the map's dimensions and the palette and style features.
    """,
    examples=("dataset finish japan --k 12 --labels folders --no-background",))
def _finish(a: FinishArgs, ctx: Context) -> Result:
    try:
        from ..relocate import dataset_health
    except ImportError:                               # pragma: no cover
        from relocate import dataset_health
    name, modality = archive.split_dataset_id(a.dataset)
    archive.open_dataset(a.dataset)
    health = dataset_health(name, modality, sample=40, **_dirs()) or {}
    if health.get("missing") and (a.feature or a.thumbnails or a.labels == "folders"):
        raise ValueError(f"{health['missing']:,} of {a.dataset}'s files are not where the index "
                         f"expects them, and building features reads them. "
                         f"`dataset relocate {a.dataset} --root <folder>` first.")
    labels = _labels(a.labels, health.get("root"))

    def compute(progress):
        return _db.rework_dataset(name, modality=modality, n_components=a.dims,
                                  features=_features(a.feature), thumbnails=a.thumbnails,
                                  k=a.k, labels=labels, progress=_db_progress(progress),
                                  should_cancel=_cancelled(progress)) or {}

    def record(tx, result):
        return _done(a.dataset, f"laid out {result.get('n_items', 0):,} items again.", result)(tx)

    return run_effect(ctx, f"dataset finish {a.dataset}", compute, record, a.background, kind="index")


class MarksArgs(Args):
    dataset: str = dataset_arg()
    background: bool = Field(False, description="Run as a job. Reading sidecars takes seconds.")


@op("dataset marks", "effect", MarksArgs, Result,
    doc="""
    Read a dataset's ON1 and XMP sidecars again. It picks up stars and colour
    tags changed since indexing, and loads no encoder and decodes no picture.
    """,
    examples=("dataset marks japan",))
def _marks(a: MarksArgs, ctx: Context) -> Result:
    name, modality = archive.split_dataset_id(a.dataset)
    archive.open_dataset(a.dataset)

    def compute(progress):
        summary = {}
        # Each layout keeps its own copy of the marks, so refresh every one.
        for n_components in (2, 3):
            try:
                summary = _db.refresh_marks(
                    name, modality=modality, n_components=n_components,
                    progress=(lambda d, t: progress(d / t if t else 0.0, f"{d:,} of {t:,}"))
                    if progress else None) or summary
            except FileNotFoundError:
                continue
        return summary

    def record(tx, summary):
        stars = summary.get("stars") or {}
        rated = sum(v for k, v in stars.items() if k)
        return _done(a.dataset, f"read {summary.get('sidecars', 0):,} sidecars, "
                                f"{rated:,} pictures rated.", summary)(tx)

    return run_effect(ctx, f"dataset marks {a.dataset}", compute, record, a.background, kind="index")


# ─────────────────────────── relocate, migrate, delete ───────────────────────────

class RelocateResult(Result):
    dataset: str
    found: int = 0
    total: int = 0
    missing: int = 0
    changed: bool = False
    dry_run: bool = False
    root: str = ""

    def text(self) -> str:
        verb = "would match" if self.dry_run else ("relocated" if self.changed else "matched")
        note = f" {self.missing:,} still missing under {self.root}." if self.missing else ""
        return f"{self.dataset}: {verb} {self.found:,} of {self.total:,} files.{note}\n"


class RelocateArgs(Args):
    dataset: str = dataset_arg()
    root: str = Field(description="The folder the files live in now.",
                      json_schema_extra={"metavar": "PATH"})
    dry_run: bool = Field(False, description="Report what would match, change nothing.")


@op("dataset relocate", "effect", RelocateArgs, RelocateResult,
    doc="""
    Point a dataset at the folder its files moved to. Files are matched by
    name and checked by content, nothing is re-encoded, and the old index is
    kept as a .bak copy.
    """,
    examples=('dataset relocate japan --root "D:/Photos/japan" --dry-run',))
def _relocate(a: RelocateArgs, ctx: Context) -> Result:
    try:
        from .. import legacy as _legacy
        from .. import relocate as _rel
    except ImportError:                               # pragma: no cover
        import legacy as _legacy
        import relocate as _rel
    name, modality = archive.split_dataset_id(a.dataset)
    if not os.path.isdir(a.root):
        raise ValueError(f"Not a folder: {a.root}")
    found = next((d for d in _discover() if d.name == name and d.modality == modality), None)
    if found is not None:
        r = _rel.relocate_legacy(found, a.root, a.dry_run, backup=True)
    else:
        bundle = next((e for e in _rel._bundle_entries() if e.get("name") == name), None)
        if bundle is None:
            raise ValueError(f"No dataset called {a.dataset} to relocate.")
        r = _rel.relocate_bundle(bundle["path"], a.root, a.dry_run)
    return RelocateResult(dataset=a.dataset, found=r.get("found", 0), total=r.get("total", 0),
                          missing=r.get("missing", 0), changed=bool(r.get("changed")),
                          dry_run=a.dry_run, root=a.root)


class MigrateArgs(Args):
    dataset: str | None = positional(0, None, description="One dataset to convert. Without it, "
                                                          "every older one.")
    thumbnails: bool = Field(False, description="Carry thumbnails into the bundle.")
    background: bool = Field(True, description="Run as a job. --no-background waits.")


@op("dataset migrate", "effect", MigrateArgs, Result,
    doc="""
    Convert older datasets to portable bundles. Bundles survive a moved folder,
    which Arcana's older pickle layout does not. The pickles stay where they
    are.
    """,
    examples=("dataset migrate japan --no-background",))
def _migrate(a: MigrateArgs, ctx: Context) -> Result:
    try:
        from .. import legacy as _legacy
    except ImportError:                               # pragma: no cover
        import legacy as _legacy
    found = _discover()
    if a.dataset:
        name, modality = archive.split_dataset_id(a.dataset)
        found = [d for d in found if d.name == name and d.modality == modality]
    if not found:
        raise ValueError("No older datasets to convert." if not a.dataset
                         else f"{a.dataset} is not an older dataset.")

    def compute(progress):
        reports = []
        for i, d in enumerate(found):
            if progress:
                progress(i / len(found), f"Converting {d.name}")
            reports.append(_legacy.convert(d, _paths_bundles(), thumbnails=a.thumbnails,
                                           verbose=False))
        return reports

    def record(tx, reports):
        lines = [f"{r.get('n_items', 0):,} items to {r.get('out')}" for r in reports]
        return _done(a.dataset or "all", "converted " + "; ".join(lines) + ".")(tx)

    return run_effect(ctx, "dataset migrate", compute, record, a.background, kind="index")


class DeleteResult(Result):
    dataset: str
    removed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    bytes: int = 0
    dry_run: bool = False

    def text(self) -> str:
        verb = "would remove" if self.dry_run else "removed"
        out = (f"{self.dataset}: {verb} {len(self.removed)} file(s), {self.bytes / 1e6:,.1f} MB. "
               f"The pictures are untouched.\n")
        out += "".join(f"  {p}\n" for p in self.removed)
        if self.failed:
            out += f"{len(self.failed)} could not be deleted (in use?):\n"
            out += "".join(f"  {p}\n" for p in self.failed)
        return out


class DeleteArgs(Args):
    dataset: str = dataset_arg()
    dry_run: bool = Field(False, description="List what would be removed, remove nothing.")


@op("dataset delete", "effect", DeleteArgs, DeleteResult,
    doc="""
    Remove a dataset's index, map and features. The pictures themselves are
    untouched. This cannot be undone, so only the person does it, and
    --dry-run lists what would go first.
    """,
    examples=("dataset delete japan --dry-run",), actors=("person", "cli"))
def _delete(a: DeleteArgs, ctx: Context) -> Result:
    try:
        from .. import legacy as _legacy
    except ImportError:                               # pragma: no cover
        import legacy as _legacy
    name, modality = archive.split_dataset_id(a.dataset)
    archive.open_dataset(a.dataset)
    r = _legacy.delete_dataset(name, modality, dry_run=a.dry_run, bundles_dir=_paths_bundles(),
                               **_dirs())
    return DeleteResult(dataset=a.dataset, removed=[str(p) for p in r.get("removed", [])],
                        failed=[str(f[0] if isinstance(f, (list, tuple)) else f)
                                for f in r.get("failed", [])],
                        bytes=int(r.get("bytes") or 0), dry_run=a.dry_run)
