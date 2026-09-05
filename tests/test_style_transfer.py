"""Applying the look of one picture to another."""

import numpy as np
import pytest
from PIL import Image, ImageFilter


def _finest_band(img) -> float:
    """Standard deviation of the finest Laplacian band -- i.e. how grainy."""
    import cv2
    a = np.asarray(img, dtype=np.float32)
    down = cv2.pyrDown(a)
    up = cv2.pyrUp(down, dstsize=(a.shape[1], a.shape[0]))
    return float((a - up).std())


@pytest.fixture
def scene():
    """A structured picture, plus grainy and smooth versions of it."""
    rng = np.random.default_rng(4)
    base = np.zeros((256, 320, 3), np.float32)
    for i in range(8):
        base[i * 32:(i + 1) * 32, :, i % 3] = 40 + i * 22
    base[:, 120:200, :] += 55
    target = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    grainy = Image.fromarray(np.clip(
        np.asarray(target, np.float32) + rng.normal(0, 24, base.shape),
        0, 255).astype(np.uint8))
    smooth = target.filter(ImageFilter.GaussianBlur(5))
    return target, grainy, smooth


def test_grain_moves_toward_the_source(scene):
    """
    A grainy source should roughen the target and a smooth one soften it.

    This is the whole point of the method, so it is worth asserting in both
    directions rather than only checking that something changed.
    """
    from arcana import style_transfer as ST
    target, grainy, smooth = scene

    base = _finest_band(target)
    rough = _finest_band(ST.texture_transfer(grainy, target, strength=1.0))
    soft = _finest_band(ST.texture_transfer(smooth, target, strength=1.0))

    assert rough > base * 1.5, f"grainy source did not roughen: {base} -> {rough}"
    assert soft < base * 0.8, f"smooth source did not soften: {base} -> {soft}"


def test_composition_survives(scene):
    """
    The target keeps its own residual, so structure and colour barely move.

    An earlier version blended the Laplacian bands directly. Those are
    spatially aligned, so it painted the source's edges at the source's
    coordinates onto a different picture -- on two portraits it ghosted one
    face over the other. Only a per-band gain crosses over now, which makes
    that impossible by construction.
    """
    from arcana import style_transfer as ST
    target, grainy, _ = scene

    out = ST.texture_transfer(grainy, target, strength=1.0)
    a = np.asarray(target, np.float32)
    b = np.asarray(out, np.float32)

    assert out.size == target.size
    # Low frequencies are the composition. Blur both hard and compare.
    la = np.asarray(target.filter(ImageFilter.GaussianBlur(8)), np.float32)
    lb = np.asarray(out.filter(ImageFilter.GaussianBlur(8)), np.float32)
    assert np.abs(la - lb).mean() < 3.0, "composition moved"
    assert abs(a.mean() - b.mean()) < 3.0, "overall brightness moved"


def test_strength_zero_is_exactly_the_target(scene):
    from arcana import style_transfer as ST
    target, grainy, _ = scene
    out = ST.texture_transfer(grainy, target, strength=0.0)
    assert np.array_equal(np.asarray(out), np.asarray(target))


def test_strength_is_monotonic(scene):
    """More strength means more of the source's grain, not a random amount."""
    from arcana import style_transfer as ST
    target, grainy, _ = scene
    vals = [_finest_band(ST.texture_transfer(grainy, target, strength=s))
            for s in (0.25, 0.5, 1.0)]
    assert vals[0] < vals[1] < vals[2], vals


def test_source_is_resized_to_the_target(scene):
    """A source of any size must work, and never change the output's size."""
    from arcana import style_transfer as ST
    target, grainy, _ = scene
    odd = grainy.resize((97, 61))
    out = ST.texture_transfer(odd, target, strength=0.8)
    assert out.size == target.size


def test_unknown_method_is_refused():
    from arcana import style_transfer as ST
    with pytest.raises(ValueError):
        ST.transfer("nonsense", None, None)


def test_transfer_dispatches_to_texture(scene):
    from arcana import style_transfer as ST
    target, grainy, _ = scene
    direct = ST.texture_transfer(grainy, target, strength=0.6)
    routed = ST.transfer("texture", grainy, target, strength=0.6)
    assert np.array_equal(np.asarray(direct), np.asarray(routed))
