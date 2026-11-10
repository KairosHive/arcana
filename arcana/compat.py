# compat.py -- the encoder calls that changed between transformers 4 and 5
#
# In transformers 4, CLIPModel.get_image_features, get_text_features and
# ClapModel.get_audio_features return the embedding as a tensor. In
# transformers 5 they return an output object with the same embedding in
# `pooler_output`. The Clap processor's `audios=` argument was renamed `audio=`.
#
# Arcana is held at transformers 4 (see pyproject.toml). These two helpers make
# every encoder call work on either, so the version can be raised when the rest
# of the app has been run against it, without touching the callers again.
# Measured on one photograph and one sound with both versions: the vectors
# agree (see VIDEO_ROADMAP.md, Status).

from __future__ import annotations


def features(out):
    """The embedding tensor from a get_*_features call, on transformers 4 or 5."""
    if hasattr(out, "detach"):
        return out
    got = getattr(out, "pooler_output", None)
    if got is None:
        raise TypeError(f"no embedding in {type(out).__name__}")
    return got


def clap_inputs(processor, audio, sampling_rate: int):
    """Clap processor inputs for a waveform, whichever name the argument has."""
    try:
        return processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")
    except (TypeError, ValueError):
        return processor(audios=audio, sampling_rate=sampling_rate, return_tensors="pt")
