"""Our own ModFlows inference path, against the checkpoint it has to load."""

import os

import numpy as np
import pytest
import torch
from PIL import Image


def _checkpoint():
    from arcana import color_transfer as ct
    p = ct.checkpoint_path()
    if not p or not os.path.exists(str(p)):
        pytest.skip("ModFlows checkpoint not downloaded")
    return p


def test_the_head_splits_exactly_into_the_velocity_field():
    """
    8195 is not arbitrary: it is every parameter of the MLP, and nothing else.

        4*1024 (W1) + 1024 (b1) + 3*1024 (W2) + 3 (b2) = 8195

    If any of those constants drifts, the split silently reshapes garbage into
    plausible-looking weights, so assert the arithmetic rather than trusting it.
    """
    from arcana import modflows_net as MN

    total = (MN.INPUT_DIM * MN.HIDDEN + MN.HIDDEN
             + MN.OUTPUT_DIM * MN.HIDDEN + MN.OUTPUT_DIM)
    assert total == MN.K_DIM == 8195

    w1, b1, w2, b2 = MN.split_params(torch.arange(MN.K_DIM, dtype=torch.float32))
    assert tuple(w1.shape) == (MN.HIDDEN, MN.INPUT_DIM)
    assert tuple(b1.shape) == (MN.HIDDEN,)
    assert tuple(w2.shape) == (MN.OUTPUT_DIM, MN.HIDDEN)
    assert tuple(b2.shape) == (MN.OUTPUT_DIM,)


def test_wrong_parameter_count_is_refused():
    from arcana import modflows_net as MN
    with pytest.raises(ValueError):
        MN.split_params(torch.zeros(8194))


def test_checkpoint_loads_strictly():
    """
    The whole reimplementation rests on this.

    The checkpoint is a state_dict keyed by torchvision's own parameter names,
    so a module built any other way would not load. strict=True is what turns
    "close enough" into a failure -- a partial load would leave a randomly
    initialised velocity field producing a plausible but wrong picture, which is
    the one failure mode that would not look like one.
    """
    from arcana import modflows_net as MN
    enc = MN.ColorFlowEncoder.from_checkpoint(_checkpoint(), "cpu")
    assert isinstance(enc, MN.ColorFlowEncoder)

    out = enc(MN.preprocess(Image.new("RGB", (64, 48), (120, 90, 60)), "cpu"))
    assert out.shape[-1] == MN.K_DIM
    assert torch.isfinite(out).all()


def test_transfer_moves_colour_toward_the_style():
    """A red target under a blue style should end up bluer than it started."""
    from arcana import modflows_net as MN

    enc = MN.ColorFlowEncoder.from_checkpoint(_checkpoint(), "cpu")
    content = Image.new("RGB", (48, 32), (200, 60, 60))
    style = Image.new("RGB", (48, 32), (60, 80, 200))

    out = MN.transfer(enc, content, style, "cpu", steps=6, strength=1.0)
    assert out.size == content.size

    before = np.asarray(content, dtype=np.float32).mean(axis=(0, 1))
    after = np.asarray(out, dtype=np.float32).mean(axis=(0, 1))
    # Blue up, red down. The exact landing point is the model's business.
    assert after[2] > before[2], f"blue did not rise: {before} -> {after}"
    assert after[0] < before[0], f"red did not fall: {before} -> {after}"


def test_strength_zero_barely_moves_the_image():
    """
    strength stops the walk early rather than scaling the field, so near zero
    the pixels have hardly travelled.
    """
    from arcana import modflows_net as MN

    enc = MN.ColorFlowEncoder.from_checkpoint(_checkpoint(), "cpu")
    content = Image.new("RGB", (32, 24), (140, 120, 100))
    style = Image.new("RGB", (32, 24), (40, 90, 190))

    a = np.asarray(MN.transfer(enc, content, style, "cpu", steps=8, strength=0.01),
                   dtype=np.float32)
    b = np.asarray(MN.transfer(enc, content, style, "cpu", steps=8, strength=1.0),
                   dtype=np.float32)
    base = np.asarray(content, dtype=np.float32)

    assert np.abs(a - base).mean() < np.abs(b - base).mean()


def test_chunking_does_not_change_the_result():
    """
    Pixels are integrated in chunks to bound memory -- a megapixel through a
    1024-wide hidden layer is about a gigabyte for that intermediate alone. Each
    pixel's path is independent, so the chunk size must not be observable.
    """
    from arcana import modflows_net as MN

    torch.manual_seed(0)
    params = MN.split_params(torch.randn(MN.K_DIM) * 0.05)
    x = torch.rand(5000, 3)

    whole = MN.integrate(params, x, steps=6, strength=1.0, chunk=10 ** 9)
    split = MN.integrate(params, x, steps=6, strength=1.0, chunk=512)
    assert torch.allclose(whole, split, atol=1e-6)


def test_preprocess_reproduces_the_trained_layout():
    """
    The reference reshapes an (H, W, 3) array to (3, H, W), which reinterprets
    the buffer instead of transposing it. That is almost certainly a mistake
    upstream, but the encoder was trained through it, so its weights only mean
    anything on inputs prepared the same way.

    This pins the quirk: a correct transpose would give a different tensor, and
    "fixing" it would silently change every result.
    """
    from arcana import modflows_net as MN

    img = Image.new("RGB", (MN.ENCODER_INPUT, MN.ENCODER_INPUT))
    img.putpixel((0, 0), (255, 0, 0))
    t = MN.preprocess(img, "cpu")
    assert tuple(t.shape) == (1, 3, MN.ENCODER_INPUT, MN.ENCODER_INPUT)

    # Under a real transpose the red pixel's channels land in three separate
    # planes at [., 0, 0]. Under the reinterpret they stay adjacent in plane 0.
    assert float(t[0, 0, 0, 0]) == pytest.approx(1.0)
    assert float(t[0, 0, 0, 1]) == pytest.approx(0.0)
    assert float(t[0, 0, 0, 2]) == pytest.approx(0.0)
