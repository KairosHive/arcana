"""
Tests for GPU capability detection and encoder-scale decoding.

Run with:  python -m pytest tests/test_gpu.py -q
       or: python tests/test_gpu.py     (no pytest needed)
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import gpu  # noqa: E402
from arcana.cvio import imread_for_encoder, imread_unicode, ENCODER_MIN_SIDE  # noqa: E402


def _reset():
    gpu._VERDICT = None


# ───────────────────────── capability probe ─────────────────────────
def test_verdict_has_a_reason_whichever_way_it_goes(tmp):
    _reset()
    v = gpu.verdict(refresh=True)
    assert v["device"] in ("cuda", "cpu")
    assert isinstance(v["ok"], bool)
    assert v.get("reason"), "a verdict must always explain itself"


def test_force_cpu_env_wins(tmp):
    _reset()
    os.environ[gpu.ENV_FORCE_CPU] = "1"
    try:
        v = gpu.verdict(refresh=True)
        assert v["ok"] is False
        assert v["device"] == "cpu"
        assert gpu.precision() == "fp32"
        assert gpu.ENV_FORCE_CPU in v["reason"]
    finally:
        del os.environ[gpu.ENV_FORCE_CPU]
        _reset()


def test_a_card_with_no_compiled_kernels_is_refused(tmp):
    """
    The silent failure this module exists for.

    torch 2.9.1+cu128 ships sm_70..sm_120 and no sm_6x. On a GTX 1080 (sm_61)
    cuda.is_available() is True, every probe in the app said yes, the encoder
    chooser promoted the user to ViT-H/14 on the strength of it, and the first
    forward pass died with "no kernel image is available" part-way through an
    index.
    """
    import torch
    if not torch.cuda.is_available():
        return                                    # nothing to fake on a CPU box

    real_cap = torch.cuda.get_device_capability
    real_name = torch.cuda.get_device_name
    torch.cuda.get_device_capability = lambda *a, **k: (6, 1)
    torch.cuda.get_device_name = lambda *a, **k: "NVIDIA GeForce GTX 1080"
    try:
        v = gpu.verdict(refresh=True)
        assert v["ok"] is False, v
        assert "sm_61" in v["reason"], v["reason"]
        assert gpu.device() == "cpu"
    finally:
        torch.cuda.get_device_capability = real_cap
        torch.cuda.get_device_name = real_name
        _reset()


def test_fp16_is_refused_below_volta(tmp):
    # .half() runs on Pascal but is slow and less accurate, and the result goes
    # into an index that is then indistinguishable from an fp32 one.
    import torch
    if not torch.cuda.is_available():
        return
    real = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *a, **k: (6, 1)
    try:
        gpu.verdict(refresh=True)
        assert gpu.use_fp16() is False
    finally:
        torch.cuda.get_device_capability = real
        _reset()


def test_probe_never_raises_even_when_torch_is_broken(tmp):
    import torch
    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: (_ for _ in ()).throw(RuntimeError("driver on fire"))
    try:
        v = gpu.verdict(refresh=True)
        assert v["ok"] is False
        assert v["device"] == "cpu"
    finally:
        torch.cuda.is_available = real
        _reset()


def test_precision_label_matches_fp16_decision(tmp):
    _reset()
    gpu.verdict(refresh=True)
    assert gpu.precision() == ("fp16" if gpu.use_fp16() else "fp32")


# ───────────────────── encoder-scale decoding ─────────────────────
def _write_jpeg(path, w, h):
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # a little structure so JPEG does not compress to nothing
    img[:, : w // 2] = (30, 60, 200)
    cv2.imwrite(path, img)
    return path


def test_large_image_is_decoded_reduced(tmp):
    p = _write_jpeg(os.path.join(tmp, "big.jpg"), 3200, 2400)
    full = imread_unicode(p)
    small = imread_for_encoder(p)
    assert full.shape[:2] == (2400, 3200)
    assert small.shape[0] < full.shape[0], small.shape
    assert min(small.shape[:2]) >= ENCODER_MIN_SIDE, small.shape


def test_small_image_is_not_shrunk_below_the_floor(tmp):
    # A 300x400 photo at 1/4 would be 75x100 -- smaller than the encoder's own
    # 224px input, which is a real loss rather than a rounding difference.
    p = _write_jpeg(os.path.join(tmp, "small.jpg"), 400, 300)
    got = imread_for_encoder(p)
    assert min(got.shape[:2]) >= min(300, ENCODER_MIN_SIDE), got.shape


def test_tiny_image_still_comes_back(tmp):
    # Below the floor at every reduction: must return the full image, not None.
    p = _write_jpeg(os.path.join(tmp, "tiny.jpg"), 120, 90)
    got = imread_for_encoder(p)
    assert got is not None
    assert got.shape[:2] == (90, 120), got.shape


def test_unreadable_file_returns_none(tmp):
    bad = os.path.join(tmp, "not-an-image.jpg")
    with open(bad, "wb") as f:
        f.write(b"certainly not a jpeg")
    assert imread_for_encoder(bad) is None
    assert imread_for_encoder(os.path.join(tmp, "missing.jpg")) is None


def test_reduced_decode_preserves_the_picture(tmp):
    # Same content, fewer pixels: the mean colour of each half must survive, or
    # the embedding would be describing a different image.
    p = _write_jpeg(os.path.join(tmp, "halves.jpg"), 2000, 1600)
    full = imread_unicode(p).astype(np.float32)
    small = imread_for_encoder(p).astype(np.float32)
    for img in (full, small):
        w = img.shape[1]
        left, right = img[:, : w // 2].mean(axis=(0, 1)), img[:, w // 2:].mean(axis=(0, 1))
        img_stats = (left, right)
        if img is full:
            f_left, f_right = img_stats
        else:
            s_left, s_right = img_stats
    assert np.abs(f_left - s_left).max() < 8, (f_left, s_left)
    assert np.abs(f_right - s_right).max() < 8, (f_right, s_right)


def test_decode_workers_is_sane(tmp):
    from arcana.db import _decode_workers
    n = _decode_workers()
    assert 2 <= n <= 24, n


# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_gpu_")
        try:
            fn(tmp)
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        import traceback
        for name, e in failed:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(e), e, e.__traceback__)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
