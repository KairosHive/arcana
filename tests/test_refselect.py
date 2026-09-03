"""
Tests for reference-colour selection.

Run with:  python -m pytest tests/test_refselect.py -q
       or: python tests/test_refselect.py     (no pytest needed)
"""

import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana.refselect import (  # noqa: E402
    analyse, filter_reference, keep_mask, palette_strip,
    _srgb_to_lab_l, _hsv_saturation,
)


def _img(pixels):
    """pixels: list of (r,g,b) repeated into a small square image."""
    n = len(pixels)
    side = int(np.ceil(np.sqrt(n)))
    buf = np.zeros((side * side, 3), dtype=np.uint8)
    for i in range(side * side):
        buf[i] = pixels[i % n]
    return Image.fromarray(buf.reshape(side, side, 3), "RGB")


def _mostly_black_with_teal(tmp, black=900, teal=100):
    return _img([(0, 0, 0)] * black + [(0, 200, 190)] * teal)


# ───────────────────────── colour maths ─────────────────────────
def test_lab_lightness_endpoints(tmp):
    black = _srgb_to_lab_l(np.array([[0, 0, 0]], dtype=np.uint8))[0]
    white = _srgb_to_lab_l(np.array([[255, 255, 255]], dtype=np.uint8))[0]
    assert abs(black - 0.0) < 0.01, black
    assert abs(white - 100.0) < 0.01, white


def test_mid_grey_is_about_53(tmp):
    # sRGB 128 is perceptually mid, which in L* is ~53.6, not 50.
    l = _srgb_to_lab_l(np.array([[128, 128, 128]], dtype=np.uint8))[0]
    assert 53.0 < l < 54.5, l


def test_saturation_of_grey_is_zero_and_pure_is_full(tmp):
    s = _hsv_saturation(np.array([[100, 100, 100], [255, 0, 0]], dtype=np.uint8))
    assert s[0] == 0.0
    assert abs(s[1] - 100.0) < 1e-4


# ───────────────────────── the mask ─────────────────────────
def test_lightness_range_excludes_black(tmp):
    px = np.array([[0, 0, 0], [0, 200, 190], [255, 255, 255]], dtype=np.uint8)
    m = keep_mask(px, l_min=20, l_max=90)
    assert not m[0], "black should be excluded"
    assert m[1], "teal should survive"
    assert not m[2], "white is above l_max"


def test_saturation_floor_excludes_grey(tmp):
    px = np.array([[120, 120, 120], [0, 200, 190]], dtype=np.uint8)
    m = keep_mask(px, s_min=25)
    assert not m[0]
    assert m[1]


def test_wide_open_mask_keeps_everything(tmp):
    px = np.array([[0, 0, 0], [255, 255, 255], [10, 90, 40]], dtype=np.uint8)
    assert keep_mask(px).all()


# ───────────────────────── the proxy ─────────────────────────
def test_wide_open_filter_returns_the_original_object(tmp):
    im = _mostly_black_with_teal(tmp)
    out, frac = filter_reference(im)
    assert out is im, "the default path must not rebuild the image"
    assert frac == 1.0


def test_filtering_out_black_shifts_the_mean_towards_teal(tmp):
    im = _mostly_black_with_teal(tmp)
    before = np.asarray(im.convert("RGB")).reshape(-1, 3).mean(axis=0)
    out, frac = filter_reference(im, l_min=15)
    after = np.asarray(out.convert("RGB")).reshape(-1, 3).mean(axis=0)
    assert 0.0 < frac < 0.5, frac
    # This is the whole point: the reference the transfer sees is now teal.
    assert after[1] > before[1] + 100, (before, after)
    assert after[2] > before[2] + 100, (before, after)


def test_proxy_contains_only_surviving_colours(tmp):
    im = _mostly_black_with_teal(tmp)
    out, _ = filter_reference(im, l_min=15)
    px = np.asarray(out).reshape(-1, 3)
    assert not (px.sum(axis=1) == 0).any(), "a black pixel leaked into the proxy"


def test_kept_fraction_is_reported(tmp):
    im = _img([(0, 0, 0)] * 750 + [(0, 200, 190)] * 250)
    _out, frac = filter_reference(im, l_min=15)
    assert 0.2 < frac < 0.3, frac


def test_empty_selection_falls_back_to_the_original(tmp):
    im = _mostly_black_with_teal(tmp)
    out, frac = filter_reference(im, l_min=99.5, l_max=100.0)
    assert frac == 0.0
    assert out is im, "an impossible filter must not produce a blank reference"


def test_proxy_is_square_and_nonempty(tmp):
    im = _mostly_black_with_teal(tmp)
    out, _ = filter_reference(im, l_min=15)
    assert out.size[0] == out.size[1]
    assert out.size[0] >= 8


def test_filter_is_deterministic(tmp):
    im = _mostly_black_with_teal(tmp)
    a, _ = filter_reference(im, l_min=15)
    b, _ = filter_reference(im, l_min=15)
    assert np.array_equal(np.asarray(a), np.asarray(b))


# ───────────────────────── the swatch strip ─────────────────────────
def test_palette_strip_returns_hex_and_shares(tmp):
    im = _mostly_black_with_teal(tmp)
    strip = palette_strip(im, n=4)
    assert strip, "expected some swatches"
    assert all(c.startswith("#") and len(c) == 7 for c, _p in strip), strip
    total = sum(p for _c, p in strip)
    assert abs(total - 1.0) < 1e-6, total


def test_palette_shares_reflect_the_picture(tmp):
    # 90% black, 10% teal -> the dark swatch must dominate the width, which is
    # the whole point of drawing them proportionally.
    im = _img([(0, 0, 0)] * 900 + [(0, 200, 190)] * 100)
    strip = palette_strip(im, n=2)
    darkest = strip[0]
    assert darkest[1] > 0.6, strip


def test_palette_strip_drops_black_when_filtered(tmp):
    im = _mostly_black_with_teal(tmp)
    filtered = palette_strip(im, n=6, l_min=15)
    assert filtered, "the teal should survive"
    for c, _p in filtered:
        r = int(c[1:3], 16); g = int(c[3:5], 16); b = int(c[5:7], 16)
        assert max(r, g, b) > 20, f"{c} is essentially black and should be gone"


def test_palette_strip_is_ordered_darkest_first(tmp):
    im = _img([(10, 10, 10), (90, 90, 90), (200, 200, 200), (250, 250, 250)] * 50)
    strip = palette_strip(im, n=4)
    ls = [_srgb_to_lab_l(np.array([[int(c[1:3], 16), int(c[3:5], 16),
                                    int(c[5:7], 16)]], dtype=np.uint8))[0]
          for c, _p in strip]
    assert ls == sorted(ls), ls


def test_palette_strip_matches_the_palette_endpoint_algorithm(tmp):
    # The strip under the sliders and the one under the Reference thumbnail
    # describe the same picture, so they must not disagree. Both go through
    # palette.extract_dominant_colors; this pins that they still agree on the
    # dominant colour of an unfiltered image.
    from arcana.palette import extract_dominant_colors
    from arcana.refselect import _lab_to_srgb
    im = _img([(0, 0, 0)] * 900 + [(0, 200, 190)] * 100)
    mine = palette_strip(im, n=2)
    bgr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
    theirs = extract_dominant_colors(bgr, n_colors=2)
    top = max(theirs, key=lambda row: row[3])
    top_hex = "#%02x%02x%02x" % _lab_to_srgb(float(top[0]), float(top[1]), float(top[2]))
    widest = max(mine, key=lambda t: t[1])[0]
    assert widest == top_hex, (widest, top_hex)


def test_dominant_colours_are_deterministic(tmp):
    # palette._get_pixels used to subsample with the unseeded global numpy RNG,
    # so extract_dominant_colors was not a function of its input: the same
    # picture produced a different palette on every call, and the two strips in
    # the moodboard disagreed with each other for no visible reason.
    from arcana.palette import extract_dominant_colors
    im = _img([(0, 0, 0)] * 400 + [(0, 200, 190)] * 300 + [(190, 40, 60)] * 300)
    bgr = np.asarray(im.convert("RGB"))[:, :, ::-1].copy()
    a = extract_dominant_colors(bgr, n_colors=3)
    b = extract_dominant_colors(bgr, n_colors=3)
    # atol rather than np.allclose's defaults. sklearn's KMeans sums over
    # chunks in parallel, so the order of floating-point reductions varies with
    # thread scheduling and successive runs differ by ~1e-5 in LAB units --
    # about 2e-5 of an 8-bit channel, which rounding to a hex swatch absorbs
    # completely. The default rtol=1e-5/atol=1e-8 fails on the a* and b*
    # channels of a neutral colour, which sit near zero, so this test was flaky
    # in roughly one full run out of three. What it needs to pin is that the
    # palette is stable, not that two float sums are bit-identical.
    assert np.allclose(a, b, atol=0.01), (a, b)


def test_unfiltered_strip_equals_the_palette_endpoint(tmp):
    # The strip under the sliders and the one under the Reference thumbnail
    # describe the same pixels, so with the filter wide open they must be the
    # same list -- not merely similar.
    from arcana.palette import extract_dominant_colors
    from arcana.refselect import _lab_to_srgb
    path = os.path.join(tmp, "ref.png")
    _img([(0, 0, 0)] * 500 + [(0, 200, 190)] * 300 + [(200, 60, 40)] * 200).save(path)

    pal = extract_dominant_colors(path, n_colors=3)
    endpoint = sorted(("#%02x%02x%02x" % _lab_to_srgb(*map(float, r[:3])))
                      for r in pal)
    mine = sorted(c for c, _p in
                  palette_strip(Image.open(path).convert("RGB"), n=3,
                                source_path=path))
    assert endpoint == mine, (endpoint, mine)


def test_lab_roundtrip_of_primaries(tmp):
    from arcana.refselect import _lab_to_srgb
    # black and white must survive the round trip exactly, or every swatch is
    # subtly wrong.
    assert _lab_to_srgb(0.0, 0.0, 0.0) == (0, 0, 0)
    assert _lab_to_srgb(100.0, 0.0, 0.0) == (255, 255, 255)


def test_palette_strip_of_impossible_filter_is_empty(tmp):
    im = _mostly_black_with_teal(tmp)
    assert palette_strip(im, l_min=99.9) == []


def test_analyse_reports_per_pixel_arrays(tmp):
    im = _mostly_black_with_teal(tmp)
    a = analyse(im)
    assert a["n"] == a["lightness"].shape[0] == a["saturation"].shape[0]
    assert a["rgb"].shape == (a["n"], 3)


def test_large_reference_is_downsampled_not_read_whole(tmp):
    big = Image.fromarray(
        np.random.RandomState(0).randint(0, 255, (2000, 3000, 3), dtype=np.uint8), "RGB")
    a = analyse(big)
    assert a["n"] <= 512 * 512, a["n"]



# ───────────────── full-resolution colour transfer ─────────────────
def test_lut3d_reproduces_a_known_colour_mapping(tmp):
    """
    A 3-D LUT must represent a transform where one output channel depends on
    another input channel. A 1-D per-channel table -- what this used to use --
    structurally cannot, which is why it scored 5.73/255 mean error against the
    flow's own output where a 65^3 table scores 1.98.
    """
    from arcana.color_transfer import _fit_lut3d, _apply_lut3d

    rng = np.random.RandomState(0)
    src = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    # swap red and blue: a pure cross-channel mapping
    dst = src[..., ::-1].copy()

    vol = _fit_lut3d(Image.fromarray(src), Image.fromarray(dst), 33)
    out = np.asarray(_apply_lut3d(Image.fromarray(src), vol), np.float32)
    err = np.abs(out - dst.astype(np.float32)).mean()
    assert err < 12.0, f"cross-channel mapping not reproduced (mean err {err:.1f})"


def test_lut3d_leaves_an_identity_mapping_alone(tmp):
    from arcana.color_transfer import _fit_lut3d, _apply_lut3d
    rng = np.random.RandomState(1)
    src = rng.randint(0, 256, (48, 48, 3), dtype=np.uint8)
    vol = _fit_lut3d(Image.fromarray(src), Image.fromarray(src), 33)
    out = np.asarray(_apply_lut3d(Image.fromarray(src), vol), np.float32)
    assert np.abs(out - src.astype(np.float32)).mean() < 6.0


def test_quality_presets_are_ordered_and_complete(tmp):
    from arcana.color_transfer import QUALITY_PRESETS, DEFAULT_QUALITY
    assert DEFAULT_QUALITY in QUALITY_PRESETS
    sizes = [QUALITY_PRESETS[k]["max_size"] for k in ("quick", "balanced", "best")]
    assert sizes == sorted(sizes), sizes
    for k, p in QUALITY_PRESETS.items():
        for field in ("max_size", "steps", "lut", "label", "note"):
            assert field in p, (k, field)
        # the LUT must be fine enough to beat the 1-D table it replaced
        assert p["lut"] >= 33, (k, p["lut"])

# ─────────────────────────── runner ───────────────────────────
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        tmp = tempfile.mkdtemp(prefix="arcana_refselect_")
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
