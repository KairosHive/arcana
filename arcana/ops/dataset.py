# dataset.py -- the read ops over datasets and encoders
#
# Creating, extending and reworking datasets are effects with jobs, and arrive
# in P4. These ops only look.

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from . import archive
from .archive import DatasetInfo, Hit, dataset_arg
from .registry import Args, Context, Result, op

try:
    from .. import core
    from .. import models as _models
except ImportError:                                   # pragma: no cover
    import core
    import models as _models


MAX_SURVEY_CLUSTERS = 16
SURVEY_ROWS_PER_SHEET = 8

# ─────────────────────────── dataset list ───────────────────────────

class DatasetListResult(Result):
    datasets: list[DatasetInfo]

    def text(self) -> str:
        if not self.datasets:
            return "No datasets yet. The person builds one from Arcana's Datasets tab.\n"
        return "\n".join(d.line() for d in self.datasets) + "\n"


class DatasetListArgs(Args):
    pass


@op("dataset list", "read", DatasetListArgs, DatasetListResult,
    doc="""
    The datasets on this machine, one line each. A line gives the size, the
    encoder, the clusters, which features (palette, style) and marks (stars,
    colour tags, dates) exist, and a warning when the files have moved since
    indexing.
    """,
    examples=("dataset list",))
def _dataset_list(a: DatasetListArgs, ctx: Context) -> DatasetListResult:
    out = []
    for d in core.find_datasets():
        ds_id = archive.dataset_id(d["name"], d["modality"])
        try:
            out.append(archive.describe_dataset(ds_id))
        except Exception as e:                        # one bad dataset must not hide the rest
            out.append(DatasetInfo(id=ds_id, modality=d["modality"], error=str(e)))
    return DatasetListResult(datasets=out)


# ─────────────────────────── dataset show ───────────────────────────

class ClusterRow(BaseModel):
    label: str
    size: int
    share: float


class DatasetShowResult(Result):
    info: DatasetInfo
    folder: str | None = None
    layouts: list[int] = Field(default_factory=list)
    clusters: list[ClusterRow] = Field(default_factory=list)
    stars: dict[str, int] = Field(default_factory=dict)
    tags: dict[str, int] = Field(default_factory=dict)

    def text(self) -> str:
        lines = [self.info.line()]
        if self.folder:
            lines.append(f"Folder: {self.folder}")
        lines.append("Layouts: " + (", ".join(f"{n}-d map" for n in self.layouts)
                                    if self.layouts else "none yet"))
        if self.clusters:
            lines.append(f"Clusters ({len(self.clusters)}), largest first:")
            width = max(len(c.label) for c in self.clusters) + 2
            lines += [f"  {c.label:<{width}}{c.size:>7,}  {c.share:.0%}" for c in self.clusters]
        if self.stars:
            lines.append("Stars: " + " · ".join(f"{k} {v:,}" for k, v in self.stars.items()))
        if self.tags:
            lines.append("Colour tags: " + " · ".join(f"{k} {v:,}" for k, v in self.tags.items()))
        return "\n".join(lines) + "\n"


class DatasetShowArgs(Args):
    dataset: str = dataset_arg()


@op("dataset show", "read", DatasetShowArgs, DatasetShowResult,
    doc="""
    One dataset in full. It gives the folder the files sit in, the layouts,
    every cluster with its size, and how many pictures carry each star rating
    and colour tag. `dataset survey` shows what the clusters look like.
    """,
    examples=("dataset show japan",))
def _dataset_show(a: DatasetShowArgs, ctx: Context) -> DatasetShowResult:
    info = archive.describe_dataset(a.dataset)
    ds = archive.open_dataset(a.dataset)
    name, modality = archive.split_dataset_id(a.dataset)
    layouts = next((d["layouts"] for d in core.find_datasets()
                    if d["name"] == name and d["modality"] == modality), [])
    paths = list(ds.idx2path.values())
    folder = None
    if paths:
        sample = paths[::max(1, len(paths) // 500)]
        try:
            folder = os.path.commonpath(sample)
        except ValueError:                            # files on several drives
            folder = None
    out = DatasetShowResult(info=info, folder=folder, layouts=layouts)
    frame = ds.frame
    if frame is not None and "label" in frame.columns:
        counts = frame["label"].astype(str).value_counts()
        total = int(counts.sum()) or 1
        out.clusters = [ClusterRow(label=str(k), size=int(v), share=int(v) / total)
                        for k, v in counts.items()]
    if frame is not None and "rating" in frame.columns and frame["rating"].notna().any():
        r = frame["rating"]
        stars = {}
        for n in range(5, -1, -1):
            c = int((r == n).sum())
            if c:
                stars[f"{n}★"] = c
        unrated = int(r.isna().sum() + (r == core.UNRATED).sum())
        if unrated:
            stars["unrated"] = unrated
        out.stars = stars
    if frame is not None and "colour" in frame.columns:
        tags = frame["colour"].astype(str)
        tags = tags[(tags != "") & (tags != "nan")]
        out.tags = {str(k): int(v) for k, v in tags.value_counts().items()}
    return out


# ─────────────────────────── dataset survey ───────────────────────────

class SurveyCluster(BaseModel):
    label: str
    size: int
    share: float
    hits: list[Hit]


class SurveyResult(Result):
    dataset: str
    total: int = 0
    n_clusters: int = 0
    clusters: list[SurveyCluster] = Field(default_factory=list)
    also: list[ClusterRow] = Field(default_factory=list)
    has_map: bool = True

    def text(self) -> str:
        if not self.has_map:
            return (f"{self.dataset} has no map yet, so there are no clusters to survey. "
                    f"search still works.\n")
        lines = [f"{self.dataset}: {self.n_clusters} clusters over {self.total:,} items, "
                 f"largest first."]
        for c in self.clusters:
            lines.append(f"\n{c.label} — {c.size:,} ({100 * c.share:.0f}%)")
            lines += ["  " + h.line() for h in c.hits]
        if self.also:
            lines.append("\nAlso: " + ", ".join(f"{c.label} ({c.size})" for c in self.also))
        return "\n".join(lines) + "\n"


class DatasetSurveyArgs(Args):
    dataset: str = dataset_arg()
    per_cluster: int = Field(4, ge=1, le=8, description="Exemplars shown per cluster.")


@op("dataset survey", "read", DatasetSurveyArgs, SurveyResult,
    doc="""
    What a dataset holds, cluster by cluster. The clusters come largest first,
    each with its most typical pictures, the ones nearest the cluster's centre.
    Do this before planning a search, so the plan fits the archive. It shows
    the sixteen largest clusters and names the rest.
    """,
    examples=("dataset survey japan", "dataset survey japan --per-cluster 2"))
def _dataset_survey(a: DatasetSurveyArgs, ctx: Context) -> SurveyResult:
    ds = archive.open_dataset(a.dataset)
    if ds.frame is None or "label" not in ds.frame.columns:
        return SurveyResult(dataset=a.dataset, has_map=False)

    keys = list(ds.idx2path.keys())
    labels = ds.frame["label"].astype(str).tolist()[:len(keys)]
    groups: dict = {}
    for k, lab in zip(keys, labels):
        groups.setdefault(lab, []).append(k)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    shown, rest = ordered[:MAX_SURVEY_CLUSTERS], ordered[MAX_SURVEY_CLUSTERS:]
    total = len(keys)

    # Exemplars: the members nearest their cluster's centre, which are what the
    # cluster is "about" rather than its outliers.
    picks = []
    for lab, members in shown:
        vecs = core.unit_rows(np.stack([ds.index.get(k) for k in members]))
        centre = core.unit(vecs.mean(axis=0))
        order = np.argsort(-(vecs @ centre))[:a.per_cluster]
        picks.append((lab, len(members), [members[i] for i in order]))

    pal = {} if ctx.sheets else archive.palettes(ds, [k for _, _, ks in picks for k in ks])
    clusters = []
    num = 0
    for lab, size, ks in picks:
        hits = []
        for k in ks:
            num += 1
            hits.append(archive.make_hit(num, a.dataset, ds, k, ds.idx2path[k], pal=pal))
        clusters.append(SurveyCluster(label=lab, size=size, share=size / total, hits=hits))
    out = SurveyResult(dataset=a.dataset, total=total, n_clusters=len(ordered),
                       clusters=clusters,
                       also=[ClusterRow(label=lab, size=len(m), share=len(m) / total)
                             for lab, m in rest])
    if ctx.sheets:
        # One cluster per row; long surveys split so no sheet is so tall the
        # client scales it down past legibility.
        for start in range(0, len(clusters), SURVEY_ROWS_PER_SHEET):
            chunk = [h for c in clusters[start:start + SURVEY_ROWS_PER_SHEET] for h in c.hits]
            out.attachments.append(archive.sheet(
                [h.path for h in chunk], [h.num for h in chunk], cols=a.per_cluster,
                caption=f"{a.dataset} survey, clusters {start + 1} to "
                        f"{min(start + SURVEY_ROWS_PER_SHEET, len(clusters))}"))
    return out


# ─────────────────────────── model list ───────────────────────────

class ModelRow(BaseModel):
    id: str
    label: str
    modality: str
    dim: int
    download_mb: int
    quality: str
    downloaded: bool


class ModelListResult(Result):
    models: list[ModelRow]

    def text(self) -> str:
        lines = []
        for m in self.models:
            state = "downloaded" if m.downloaded else f"{m.download_mb:,} MB to download"
            lines.append(f"{m.label}  ({m.modality}, {m.dim}-d, {m.quality}, {state})")
            lines.append(f"  {m.id}")
        return "\n".join(lines) + "\n"


class ModelListArgs(Args):
    modality: Literal["image", "audio", "all"] = Field("all", description="Which encoders to list.")


@op("model list", "read", ModelListArgs, ModelListResult,
    doc="""
    The encoders Arcana can index with. They come cheapest first, with their
    download size, the dimension of the vectors they make, and whether they
    are already downloaded. A dataset's encoder is fixed when it is indexed.
    """,
    examples=("model list", "model list --modality image"))
def _model_list(a: ModelListArgs, ctx: Context) -> ModelListResult:
    mods = ["image", "audio"] if a.modality == "all" else [a.modality]
    rows = []
    for m in mods:
        for d in _models.catalogue(m, has_gpu=False):
            rows.append(ModelRow(id=d["id"], label=d["label"], modality=d["modality"],
                                 dim=d["dim"], download_mb=d["download_mb"],
                                 quality=d["quality"], downloaded=d["downloaded"]))
    return ModelListResult(models=rows)
