"""
The colour transfer registry and the two classical methods.

Each method leaves the content as it was at strength 0 and when the reference
is the content itself. MKL's output carries the reference's colour covariance.
IDT's output carries the reference's distribution along every axis, not only
its mean and spread. Every registry id dispatches, and `colour methods` lists
exactly the registry.

Run with:  python -m pytest tests/test_transfer_methods.py -q
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import classic_transfer as ct, transfer_methods as tm  # noqa: E402


def _picture(seed, shape=(96, 128), tint=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.18, (*shape, 3)) + np.array(tint)
    # Some structure, so channels are correlated the way photographs are.
    base[..., 1] = 0.6 * base[..., 0] + 0.4 * base[..., 1]
    return Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))


CONTENT = _picture(1, tint=(0.1, -0.05, -0.1))
REFERENCE = _picture(2, tint=(-0.15, 0.05, 0.2))


def _arr(img):
    return np.asarray(img, np.float64) / 255.0


CLASSIC = ("lab", "mkl", "idt")


@pytest.mark.parametrize("method", CLASSIC)
def test_strength_zero_leaves_the_content(method):
    out, _ = tm.transfer(method, CONTENT, REFERENCE, strength=0.0)
    assert np.abs(_arr(out) - _arr(CONTENT)).mean() < 2 / 255


@pytest.mark.parametrize("method", ("mkl", "idt"))
def test_the_content_as_its_own_reference_barely_moves(method):
    out, _ = tm.transfer(method, CONTENT, CONTENT, strength=1.0)
    assert np.abs(_arr(out) - _arr(CONTENT)).mean() < 4 / 255


def _cloud_picture(seed, mean, cov, shape=(120, 160)):
    rng = np.random.default_rng(seed)
    px = rng.multivariate_normal(mean, cov, shape[0] * shape[1])
    return Image.fromarray((np.clip(px, 0, 1) * 255).astype(np.uint8).reshape(*shape, 3))


def test_mkl_carries_the_reference_covariance():
    # Two pictures whose colour covariances differ in spread and correlation,
    # kept well inside 0..1 so clipping does not blur the comparison.
    content = _cloud_picture(5, [0.45, 0.5, 0.5], [[0.010, 0.006, 0], [0.006, 0.010, 0],
                                                    [0, 0, 0.004]])
    reference = _cloud_picture(6, [0.55, 0.45, 0.4], [[0.004, -0.002, 0], [-0.002, 0.006, 0.001],
                                                      [0, 0.001, 0.012]])
    out = ct.mkl(content, reference, 1.0)
    got = np.cov(_arr(out).reshape(-1, 3), rowvar=False)
    want = np.cov(_arr(reference).reshape(-1, 3), rowvar=False)
    before = np.cov(_arr(content).reshape(-1, 3), rowvar=False)
    assert np.abs(got - want).max() < 0.15 * np.abs(before - want).max()
    assert np.allclose(_arr(out).reshape(-1, 3).mean(0), _arr(reference).reshape(-1, 3).mean(0),
                       atol=0.01)


def test_mkl_matrix_is_exact_on_gaussian_clouds():
    rng = np.random.default_rng(3)
    a = rng.multivariate_normal([0.2, 0.3, 0.4], [[0.02, 0.01, 0], [0.01, 0.03, 0.005],
                                                  [0, 0.005, 0.01]], 20000)
    b = rng.multivariate_normal([0.6, 0.5, 0.3], [[0.01, -0.004, 0], [-0.004, 0.02, 0],
                                                  [0, 0, 0.04]], 20000)
    t, mu_a, mu_b = ct.mkl_matrix(a, b)
    moved = (a - mu_a) @ t + mu_b
    assert np.allclose(np.cov(moved, rowvar=False), np.cov(b, rowvar=False), atol=1e-3)


def test_idt_matches_every_marginal_after_its_iterations():
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 1, (6000, 3)) ** 2
    b = np.column_stack([rng.beta(5, 2, 6000), rng.beta(2, 5, 6000), rng.normal(0.5, 0.1, 6000)])
    moved = ct.idt_cloud(a, b, iterations=20)
    for axis in np.eye(3):
        qa = np.quantile(moved @ axis, [0.1, 0.5, 0.9])
        qb = np.quantile(b @ axis, [0.1, 0.5, 0.9])
        assert np.abs(qa - qb).max() < 0.03, (axis, qa, qb)


@pytest.mark.parametrize("method", CLASSIC)
def test_every_classic_method_keeps_the_size(method):
    big = CONTENT.resize((600, 450))
    out, info = tm.transfer(method, big, REFERENCE)
    assert out.size == big.size and info["label"] == tm.BY_ID[method].label


def test_the_registry_is_ordered_cheapest_first_and_complete():
    assert [m.id for m in tm.METHODS] == ["lab", "mkl", "idt", "modflows"]
    cpu = [m.cpu_ms for m in tm.METHODS]
    assert cpu[:3] == sorted(cpu[:3])
    for m in tm.METHODS:
        assert m.blurb and m.label and m.params and "strength" in m.params
    assert tm.BY_ID["modflows"].params == ("strength", "quality")


def test_an_unknown_method_is_refused_with_the_list():
    ok, why = tm.available("vibes")
    assert not ok and "lab, mkl, idt, modflows" in why
    with pytest.raises(ValueError):
        tm.transfer("vibes", CONTENT, REFERENCE)


def test_colour_methods_lists_exactly_the_registry():
    from arcana import ops
    rows = ops.call("colour methods").methods
    assert [r.id for r in rows] == [m.id for m in tm.METHODS]
    assert all(r.available for r in rows if r.id in CLASSIC)
