# on1.py -- read ON1 Photo RAW sidecars
#
# ON1 stores its per-photo decisions beside the photograph, in a `.on1` file
# sharing the basename:
#
#   DSC06800.JPG
#   DSC06800.on1
#
# The sidecar is JSON, one object per photograph keyed by ON1's own GUID:
#
#   {"photos": {"<guid>": {"name": "DSC06800.JPG",
#                          "metadata": {"Rating": 4, "Label": "Yellow",
#                                       "CaptureDate": "Sat Oct 28 14:26:25 2023",
#                                       ...}}},
#    "type": 1, "version": 2026}
#
# Two details the obvious reader gets wrong:
#
#   * A sidecar can describe more than one file. Shooting RAW+JPEG produces
#     DSC06800.ARW and DSC06800.JPG, which share a basename and therefore share
#     one sidecar with two entries. 4,114 of the 55,783 sidecars in the archive
#     this was written against are such pairs, so taking the first entry gets the
#     wrong file's marks about half the time. Match on `name`.
#   * `Rating` and `Label` are absent, not 0 and not "", for a photograph ON1
#     has never been shown. Absent and zero mean the same thing to a filter, but
#     conflating them loses the ability to say "no opinion recorded".
#
# This module only reads. Writing marks back out is a separate job with its own
# risks -- ON1 owns those files and is often running while Arcana is.

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

SUFFIX = ".on1"
XMP_SUFFIX = ".xmp"

# ON1's colour tags, in the order its own filter rail shows them, so the UI has
# a stable ordering. A tag not on this list is still read and still shown.
LABELS = ("Red", "Yellow", "Green", "Blue", "Purple")

# ON1 writes something close to C's asctime: "Sat Oct 28 14:26:25 2023". Close,
# because its Thursday is "Thur", which no locale defines and %a therefore
# rejects -- 7,739 of the 55,783 sidecars measured here, every one of which
# would silently lose its capture date. The weekday is redundant with the date
# anyway, so it is dropped before parsing rather than matched. Days are padded
# with a space by some versions and a zero by others, so whitespace is collapsed
# first and %d then accepts both.
_DATE_FORMATS = ("%b %d %H:%M:%S %Y", "%Y-%m-%dT%H:%M:%S")
_WEEKDAY = re.compile(r"^[A-Za-z]{3,5}\s+")


@dataclass
class Marks:
    """What is known about one photograph from outside Arcana. Every field may be unset."""

    rating: int | None = None       # 0-5 stars; 0 means "seen, not starred"
    label: str = ""                 # colour tag, "" when none
    flag: int | None = None         # ON1's pick/reject flag, rarely set
    captured: str = ""              # ISO-8601 capture time in the camera's own zone
    camera: str = ""                # "SONY ILCE-7M3", for display
    source: str = ""                # "on1" | "exif" -- where this came from

    def is_empty(self) -> bool:
        return (self.rating is None and not self.label
                and self.flag is None and not self.captured and not self.camera)

    def to_dict(self) -> dict:
        """Only the fields that are set, so bundles do not carry empty keys."""
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


def sidecar_for(media_path: str) -> str:
    """Path of the sidecar that would describe `media_path`, present or not."""
    return os.path.splitext(media_path)[0] + SUFFIX


def xmp_for(media_path: str) -> str:
    """Path of the Adobe-format sidecar for `media_path`, present or not."""
    return os.path.splitext(media_path)[0] + XMP_SUFFIX


# XMP namespaces, only the ones carrying a decision or a date.
_XMP_NS = "http://ns.adobe.com/xap/1.0/"
_EXIF_NS = "http://ns.adobe.com/exif/1.0/"
_TIFF_NS = "http://ns.adobe.com/tiff/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# XMP stores the colour tag as a name, but which names depend on the writer's
# locale and version; lower-casing and mapping back to ON1's own vocabulary is
# what keeps one filter working across both sidecar formats.
_XMP_LABELS = {name.lower(): name for name in LABELS}


def _read_xmp(path: str) -> "Marks | None":
    """
    Marks from an Adobe-format sidecar.

    ON1 writes `.xmp` as well as `.on1` in some configurations, and Lightroom
    writes only `.xmp`. Reading both means a rating survives whichever tool the
    photographer used, which is the entire point of deferring to it.
    """
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError, ValueError):
        return None
    desc = root.find(f".//{{{_RDF_NS}}}Description")
    if desc is None:
        return None

    def attr(ns, key):
        return desc.attrib.get(f"{{{ns}}}{key}")

    rating = attr(_XMP_NS, "Rating")
    try:
        rating = int(float(rating)) if rating not in (None, "") else None
    except ValueError:
        rating = None
    # Lightroom writes -1 for "rejected". That is not a star count, and letting
    # it through would make a "1 star and up" filter sort rejects above unrated.
    if rating is not None and rating < 0:
        rating = None

    label = (attr(_XMP_NS, "Label") or "").strip()
    marks = Marks(
        rating=rating,
        label=_XMP_LABELS.get(label.lower(), label),
        captured=_parse_date(attr(_EXIF_NS, "DateTimeOriginal")),
        camera=_camera(attr(_TIFF_NS, "Make"), attr(_TIFF_NS, "Model")),
        source="xmp",
    )
    return None if marks.is_empty() else marks


def _parse_date(raw, offset_seconds=None) -> str:
    """ON1's asctime string -> ISO-8601, keeping the camera's own local time."""
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = " ".join(raw.split())
    # Only when a month name follows, so an ISO string is left intact.
    if _WEEKDAY.match(cleaned) and not cleaned[:4].isdigit():
        cleaned = _WEEKDAY.sub("", cleaned, count=1)
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        # CaptureDateOffset is the camera's UTC offset in seconds. Keeping it
        # makes "shot at 19:15" stay true across an archive that spans time
        # zones, which a travel library always does.
        if isinstance(offset_seconds, (int, float)):
            try:
                dt = dt.replace(tzinfo=timezone(timedelta(seconds=int(offset_seconds))))
            except ValueError:
                pass
        return dt.isoformat()
    return ""


def _entry_for(doc: dict, media_path: str) -> dict | None:
    """The photo record inside `doc` that describes `media_path`."""
    photos = doc.get("photos")
    if not isinstance(photos, dict) or not photos:
        return None
    wanted = os.path.basename(media_path).lower()
    for rec in photos.values():
        if isinstance(rec, dict) and str(rec.get("name", "")).lower() == wanted:
            return rec
    # A lone entry whose name does not match is still almost certainly about
    # this file: ON1 records the name at the time it last saw it, and the file
    # may have been renamed since. With two or more entries there is no way to
    # tell which is meant, so decline rather than guess.
    if len(photos) == 1:
        only = next(iter(photos.values()))
        return only if isinstance(only, dict) else None
    return None


def read_marks(media_path: str, *, exif_fallback: bool = True) -> Marks | None:
    """
    Marks for one media file, or None when nothing is known about it.

    Reads the sidecar when there is one. With `exif_fallback`, a file ON1 has
    never touched still yields its capture date from its own EXIF header, so a
    date filter covers the whole collection rather than only the culled part.
    """
    doc = None
    path = sidecar_for(media_path)
    if os.path.exists(path):
        try:
            # utf-8-sig, because ON1 writes a BOM on some platforms and the
            # strict decoder rejects the file outright rather than skipping it.
            with open(path, "r", encoding="utf-8-sig") as fh:
                doc = json.load(fh)
        except (OSError, ValueError, UnicodeDecodeError):
            doc = None

    if isinstance(doc, dict):
        rec = _entry_for(doc, media_path)
        if rec is not None:
            meta = rec.get("metadata") or {}
            rating = meta.get("Rating")
            flag = meta.get("UserFlag")
            marks = Marks(
                rating=int(rating) if isinstance(rating, (int, float)) else None,
                label=str(meta.get("Label") or "").strip(),
                flag=int(flag) if isinstance(flag, (int, float)) else None,
                captured=_parse_date(meta.get("CaptureDate"), meta.get("CaptureDateOffset")),
                camera=_camera(meta.get("CameraMake"), meta.get("CameraModel")),
                source="on1",
            )
            if marks.captured or not exif_fallback:
                return marks
            # A sidecar without a usable CaptureDate still has stars worth
            # keeping; take only the date from EXIF and leave the rest alone.
            exif = _exif_marks(media_path)
            if exif is not None:
                marks.captured = exif.captured
                marks.camera = marks.camera or exif.camera
            return marks

    xmp_path = xmp_for(media_path)
    if os.path.exists(xmp_path):
        marks = _read_xmp(xmp_path)
        if marks is not None:
            if marks.captured or not exif_fallback:
                return marks
            exif = _exif_marks(media_path)
            if exif is not None:
                marks.captured = exif.captured
                marks.camera = marks.camera or exif.camera
            return marks

    return _exif_marks(media_path) if exif_fallback else None


def _camera(make, model) -> str:
    make = str(make or "").strip()
    model = str(model or "").strip()
    # ON1 writes "----" for a lens or body it could not identify.
    parts = [x for x in (make, model) if x and set(x) != {"-"}]
    return " ".join(parts)


# EXIF tag ids, so no tag-name table has to be built per file.
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME = 306
_EXIF_MAKE = 271
_EXIF_MODEL = 272
_EXIF_SUB_IFD = 0x8769


def _exif_marks(media_path: str) -> Marks | None:
    """Capture date and camera from the file's own EXIF, or None."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(media_path) as im:
            exif = im.getexif()
            if not exif:
                return None
            raw = exif.get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
            if not raw:
                try:
                    raw = (exif.get_ifd(_EXIF_SUB_IFD) or {}).get(_EXIF_DATETIME_ORIGINAL)
                except (AttributeError, KeyError, OSError, ValueError):
                    raw = None
            camera = _camera(exif.get(_EXIF_MAKE), exif.get(_EXIF_MODEL))
    except (OSError, ValueError, TypeError, AttributeError, MemoryError):
        return None

    # EXIF writes "2023:10:28 14:26:25" -- colons in the date half too.
    captured = ""
    if isinstance(raw, str) and len(raw) >= 19:
        head, _, tail = raw.strip().partition(" ")
        captured = _parse_date(head.replace(":", "-") + "T" + tail)
    marks = Marks(captured=captured, camera=camera, source="exif" if captured else "")
    return None if marks.is_empty() else marks


def scan(paths, *, workers: int = 16, exif_fallback: bool = True,
         progress=None) -> dict[str, "Marks"]:
    """
    Marks for many files, keyed by the path given.

    A sidecar is one stat plus one small read, so this is I/O-bound and threads
    help despite the GIL: on an external drive, scanning 55,783 sidecars with 16
    threads is minutes rather than tens of minutes.

    progress: callable(done, total)
    """
    paths = list(paths)
    out: dict[str, Marks] = {}
    if not paths:
        return out

    def one(p):
        try:
            return p, read_marks(p, exif_fallback=exif_fallback)
        except Exception:
            return p, None

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for p, marks in pool.map(one, paths):
            done += 1
            if marks is not None and not marks.is_empty():
                out[p] = marks
            if progress is not None and (done % 500 == 0 or done == len(paths)):
                progress(done, len(paths))
    return out


def summarise(marks: dict) -> dict:
    """Counts for a one-line report: how much of a dataset ON1 has an opinion about."""
    stars: dict[int, int] = {}
    labels: dict[str, int] = {}
    dated = sidecars = 0
    for m in marks.values():
        if m.rating is not None:
            stars[m.rating] = stars.get(m.rating, 0) + 1
        if m.label:
            labels[m.label] = labels.get(m.label, 0) + 1
        if m.captured:
            dated += 1
        if m.source == "on1":
            sidecars += 1
    return {"n": len(marks), "sidecars": sidecars, "dated": dated,
            "stars": dict(sorted(stars.items())),
            "labels": dict(sorted(labels.items(), key=lambda kv: -kv[1]))}
