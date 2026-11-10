# modality.py -- the kinds of media Arcana knows, written once
#
# A dataset holds one kind of media: photos, sounds or videos. That kind decides
# which encoder indexes it, how a result is drawn and which features work on it.
# Before this module the two words "image" and "audio" were spelled out in the
# dataset id suffix (three copies), in the regexes that find dataset files (two
# copies) and in four sets of file extensions. Everything about a modality as a
# name lives here now, and this module imports nothing from Arcana, so every
# other module can use it.
#
# What each feature can do with each modality is declared where the feature is:
# on the op (ops/registry.py, `modalities=`) and in capabilities.py for what the
# page draws. See VIDEO_ROADMAP.md, part 1.

from __future__ import annotations

import os
import re

IMAGE = "image"
AUDIO = "audio"
VIDEO = "video"

# Every modality, in the order tags are shown.
ALL: tuple[str, ...] = (IMAGE, AUDIO, VIDEO)

# What a person reads. The id stays "image" because every index file on disk is
# named with it; the word on screen is "photo".
LABEL = {IMAGE: "photo", AUDIO: "audio", VIDEO: "video"}
NOUN = {IMAGE: "picture", AUDIO: "sound", VIDEO: "clip"}
NOUNS = {IMAGE: "pictures", AUDIO: "sounds", VIDEO: "clips"}
PLURAL = {IMAGE: "photos", AUDIO: "sounds", VIDEO: "videos"}

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
AUDIO_EXTS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
EXTS = {IMAGE: IMAGE_EXTS, AUDIO: AUDIO_EXTS, VIDEO: VIDEO_EXTS}


def is_modality(value) -> bool:
    return value in ALL


def encoder_of(modality: str | None) -> str:
    """
    The modality whose encoders embed this one.

    A video section is embedded by the image encoders (CLIP reads its frames),
    so wherever the code picks an encoder, a text tower or a label vocabulary,
    video follows image. Sound has its own.
    """
    return AUDIO if modality == AUDIO else IMAGE


# ─────────────────────────── a section of a video, as a path ───────────────────────────
# Every index maps a key to a path, and the whole app (search, the map, cards,
# boards, handles) passes those paths around. A video dataset has many keys per
# file, one per section, so a section is written as its file plus a time range
# in the form a browser already understands as a media fragment:
#
#     C:\footage\walk.mp4#t=12.000,22.500
#
# That keeps one string per key everywhere, and a <video> element given the
# same fragment plays exactly that range.

FRAGMENT = "#t="


def section_ref(path: str, t0: float, t1: float) -> str:
    return f"{path}{FRAGMENT}{t0:.3f},{t1:.3f}"


def split_ref(ref: str) -> tuple[str, float | None, float | None]:
    """(file, t0, t1). The times are None for a plain path."""
    ref = ref or ""
    file, sep, times = ref.rpartition(FRAGMENT)
    if not sep:
        return ref, None, None
    try:
        a, _, b = times.partition(",")
        return file, float(a), (float(b) if b else None)
    except ValueError:
        return ref, None, None


def file_of(ref: str) -> str:
    """The file a reference names, with any time range taken off."""
    return split_ref(ref)[0]


def of_path(path: str) -> str | None:
    """The modality of a file, from its extension. None for anything else."""
    ext = os.path.splitext(file_of(path or ""))[1].lower()
    for modality, exts in EXTS.items():
        if ext in exts:
            return modality
    return None


# ─────────────────────────── dataset ids ───────────────────────────
# A photo dataset is named by its bare name (japan), every other kind by its
# name and modality (field@audio, rushes@video). The bare form stays for photos
# because every board, skill and habit written so far uses it.

def dataset_id(name: str, modality: str | None) -> str:
    modality = modality or IMAGE
    return name if modality == IMAGE else f"{name}@{modality}"


def split_dataset_id(ds_id: str) -> tuple[str, str]:
    name, sep, tail = (ds_id or "").rpartition("@")
    if sep and name and tail in ALL and tail != IMAGE:
        return name, tail
    return ds_id, IMAGE


def of_handle(handle: str) -> str | None:
    """The modality a handle (japan:1423, field@audio:12) names, or None when it is not one."""
    ds_id, sep, key = (handle or "").strip().rpartition(":")
    if not sep or not ds_id or not key.isdigit():
        return None
    return split_dataset_id(ds_id)[1]


# ─────────────────────────── dataset files ───────────────────────────
# One pattern for each kind of dataset file, so a new modality is found
# everywhere at once.

_ALT = "|".join(ALL)
INDEX_RE = re.compile(rf"index_(.+)_({_ALT})\.pkl$")
LATENT_RE = re.compile(rf"latent_space_(.+)_({_ALT})_(\d+)d\.pkl$")


def feature_file(name: str, kind: str, modality: str | None = IMAGE) -> str:
    """
    The file name of a feature block (palette, style, motion) of a dataset.

    Photo datasets keep the name they have always had, features_<name>_<kind>.npz,
    so nothing on disk has to move. Every other modality carries its own name,
    which is what keeps a photo set and an audio set called the same thing from
    reading each other's features.
    """
    modality = modality or IMAGE
    if modality == IMAGE:
        return f"features_{name}_{kind}.npz"
    return f"features_{name}_{modality}_{kind}.npz"


# ─────────────────────────── saying it ───────────────────────────

def words(modalities, plural: bool = True) -> str:
    """'photos', 'photos and sounds', 'photos, sounds and videos', in the fixed order."""
    table = PLURAL if plural else LABEL
    names = [table[m] for m in ALL if m in set(modalities)]
    if not names:
        return "nothing"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def tags(modalities) -> list[str]:
    """The modalities as ids, in the fixed order, for the page and the schema."""
    return [m for m in ALL if m in set(modalities)]
