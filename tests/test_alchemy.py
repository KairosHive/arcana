"""
Alchemy: the catalogue, the masks, the finish, the recipes and the ops, without
loading a model.

The token and vision models are exercised by hand against real photographs.
Here every check runs on synthetic pictures and on the methods that need no
weights, so the suite stays fast and runs on a machine without a GPU.

Run with:  python -m pytest tests/test_alchemy.py -q
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import ops  # noqa: E402
from arcana.alchemy import finish, mask, methods, recipes, tokens  # noqa: E402
from arcana.ops import registry  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    out = synth.use(tmp, monkeypatch)
    monkeypatch.setitem(registry.FLAG_PROBES, "gpu", lambda: False)
    return out


def _picture(h=96, w=128):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / w, y / h, 0.5 + 0.5 * np.sin(x / 9.0)], -1)
    return np.clip(img, 0, 1).astype(np.float32)


# ─────────────────────────── the catalogue ───────────────────────────

def test_every_method_declares_well_formed_knobs():
    assert set(methods.FAMILIES) == {"pixels", "tokens", "colour", "generative"}
    for m in methods.METHODS.values():
        assert m.id.startswith(m.family + "."), m.id
        assert m.doc and m.label
        keys = [k.key for k in m.knobs]
        assert len(keys) == len(set(keys)), m.id
        for k in m.knobs:
            if k.kind in ("float", "int", "angle"):
                assert k.lo is not None and k.hi is not None and k.lo <= k.default <= k.hi, (m.id, k.key)
            if k.kind == "choice":
                assert k.default in k.options, (m.id, k.key)
            for key, _, _ in k.when:
                assert key in keys, (m.id, k.key, key)


def test_knobs_are_brought_into_range_and_choices_checked():
    m = methods.get("tokens.mix")
    got = methods.settle_knobs(m, {"push": 99, "seam": -4})
    assert got["push"] == 3 and got["seam"] == 0
    with pytest.raises(ValueError, match="model is one of"):
        methods.settle_knobs(m, {"model": "nope"})
    with pytest.raises(ValueError, match="No method"):
        methods.get("tokens.nope")


def test_weights_are_shares_among_the_materials():
    assert list(tokens.weights_of([0.8, 0.8], 2)) == [0.5, 0.5]
    assert list(tokens.weights_of([3.0], 1)) == [1.0]


def test_every_method_leads_with_its_dominance_knob():
    """The knob deciding how much the picture or the materials dominate comes first in main."""
    lead = {"pixels": "amount", "colour": "strength"}
    for m in methods.METHODS.values():
        assert {k.section for k in m.knobs} <= set(methods.SECTIONS), m.id
        if m.family in lead:
            assert m.knobs[0].key == lead[m.family], m.id
    assert methods.get("tokens.mix").knobs[0].key == "picture"
    assert methods.get("tokens.mosh").knobs[0].key == "hold"


def test_pixel_amount_lays_the_result_over_the_picture():
    import types
    c = _picture()
    t = c[::-1].copy()
    step = types.SimpleNamespace(c=c, materials=[t], H=c.shape[0], W=c.shape[1])
    m = methods.get("pixels.chroma")
    whole = m.run(step, methods.settle_knobs(m, {}))
    none = m.run(step, methods.settle_knobs(m, {"amount": 0}))
    half = m.run(step, methods.settle_knobs(m, {"amount": 0.5}))
    assert np.allclose(none, c, atol=1e-6)
    assert np.allclose(half, (c + whole) / 2, atol=1e-5)


# ─────────────────────────── masks ───────────────────────────

def test_every_procedural_preset_builds_a_mask_in_range():
    c = _picture()
    sc = mask.Scene(c, "synthetic", material=c[::-1].copy())
    for p in mask.PRESETS:
        if p.source.split(":")[0] in ("subject", "background", "thing", "stuff", "prompt", "words") \
                or "depth" in p.source or p.group == "depth and focus":
            continue                                   # these ask a vision model
        m = mask.build({"layers": [p.name]}, sc)
        assert m.shape == c.shape[:2], p.name
        assert 0.0 <= m.min() and m.max() <= 1.0, p.name


def test_coverage_and_combining():
    c = _picture()
    sc = mask.Scene(c, "synthetic")
    a = mask.build({"layers": ["highlights"], "coverage": [0.3], "feather": 0}, sc)
    assert abs(float((a > 0.5).mean()) - 0.3) < 0.08
    top = mask.build({"layers": ["from the top"], "feather": 0}, sc)
    both = mask.build({"layers": ["highlights", "from the top"], "combine": ["and"], "coverage": [0.3],
                       "feather": 0}, sc)
    either = mask.build({"layers": ["highlights", "from the top"], "combine": ["or"], "coverage": [0.3],
                         "feather": 0}, sc)
    assert both.mean() <= min(a.mean(), top.mean()) + 1e-3
    assert either.mean() >= max(a.mean(), top.mean()) - 1e-3
    inv = mask.build({"layers": ["highlights"], "coverage": [0.3], "invert": True, "feather": 0}, sc)
    assert abs(float(inv.mean() + a.mean()) - 1.0) < 0.05
    assert mask.build({"layers": ["none"]}, sc) is None


# ─────────────────────────── finish ───────────────────────────

def test_finish_at_rest_leaves_the_picture():
    c = _picture()
    assert finish.Finish().rest
    assert np.allclose(finish.apply(c, {}, None), c, atol=1e-6)


def test_finish_moves_light_the_right_way_and_honours_its_scope():
    c = _picture() * 0.6 + 0.2
    brighter = finish.apply(c, {"exposure": 0.5}, None)
    assert brighter.mean() > c.mean()
    warmer = finish.apply(c, {"temperature": 50}, None)
    assert (warmer[..., 0] - warmer[..., 2]).mean() > (c[..., 0] - c[..., 2]).mean()
    m = np.zeros(c.shape[:2], np.float32)
    m[:, : c.shape[1] // 2] = 1
    inside = finish.apply(c, {"exposure": 0.5, "scope": "mask"}, m)
    right = slice(c.shape[1] // 2 + 8, None)
    assert np.allclose(inside[:, right], c[:, right], atol=1e-4)


# ─────────────────────────── recipes ───────────────────────────

def test_builtin_recipes_name_real_methods_knobs_and_presets():
    names = [r.name for r in recipes.BUILTIN]
    assert len(names) == len(set(names))
    for r in recipes.BUILTIN:
        assert set(r.settings) <= set(recipes.SETTING_KEYS), r.name
        if "method" in r.settings:
            m = methods.get(r.settings["method"])
            methods.settle_knobs(m, r.settings.get("knobs") or {})
            assert set(r.settings.get("knobs") or {}) <= {k.key for k in m.knobs}, r.name
        for line in (r.settings.get("mask") or {}).get("layers", []):
            assert line in mask.PRESET, (r.name, line)
        assert set(r.settings.get("finish") or {}) <= set(finish.Finish.__dataclass_fields__), r.name


def test_a_recipe_sets_only_what_it_holds_and_given_settings_win():
    r = recipes.get("water light")
    spec = recipes.apply(r, {"knobs": {"push": 0.5}, "finish": {"exposure": 0.2}})
    assert spec["method"] == "tokens.mix"
    assert spec["knobs"]["push"] == 0.5 and spec["knobs"]["carry"] == "transport"
    assert spec["finish"] == {"exposure": 0.2}


def test_saved_recipes_are_listed_and_replace_their_namesake(tmp, monkeypatch):
    monkeypatch.setattr(recipes, "_store", lambda: os.path.join(tmp, "recipes.json"))
    recipes.save("mine", {"finish": {"exposure": 0.1}, "picture": "ignored"})
    recipes.save("mine", {"finish": {"exposure": 0.3}})
    mine = [r for r in recipes.every() if r.name == "mine"]
    assert len(mine) == 1 and mine[0].own and mine[0].settings == {"finish": {"exposure": 0.3}}


# ─────────────────────────── ops ───────────────────────────

def test_the_ops_are_registered_with_their_kinds():
    kinds = {o.name: o.kind for o in ops.all_ops() if o.name.startswith("alchemy ")}
    assert kinds == {"alchemy methods": "read", "alchemy memory": "read", "alchemy preview": "read",
                     "alchemy make": "effect",
                     "alchemy mask presets": "read", "alchemy mask preview": "read",
                     "alchemy recipe list": "read", "alchemy recipe show": "read",
                     "alchemy recipe save": "write", "alchemy film list": "read",
                     "alchemy film show": "read", "alchemy film save": "write", "alchemy film delete": "write",
                     "alchemy film preview": "read", "alchemy film make": "effect"}


def test_settings_are_routed_and_strays_refused():
    from arcana.ops.alchemy import AlchemyPreviewArgs, settings_of
    a = AlchemyPreviewArgs(picture="x:1", material=["x:2"], method="pixels.displace",
                           set=["amp=0.05", "finish.exposure=0.3", "mask.feather=9"], mask=["sky"])
    spec = settings_of(a)
    assert spec["knobs"] == {"amp": 0.05}
    assert spec["finish"] == {"exposure": 0.3}
    assert spec["mask"] == {"feather": 9, "layers": ["sky"]}
    with pytest.raises(ValueError, match="no knob called 'bogus'"):
        settings_of(AlchemyPreviewArgs(method="pixels.displace", set=["bogus=1"]))
    with pytest.raises(ValueError, match="No finish setting"):
        settings_of(AlchemyPreviewArgs(set=["finish.bogus=1"]))
    with pytest.raises(ValueError, match="no mask refinement"):
        settings_of(AlchemyPreviewArgs(set=["mask.bogus=1"]))


def test_preview_and_make_on_the_synthetic_dataset(tmp, ds):
    r = ops.call("alchemy preview", {"picture": "synth:12", "material": ["synth:7"],
                                     "method": "pixels.displace", "mask": ["highlights"], "size": 256})
    assert r.method == "pixels.displace" and r.attachments and r.mask_share is not None
    with pytest.raises(ops.Refusal, match="needs at least 1 material"):
        ops.call("alchemy preview", {"picture": "synth:12", "method": "colour.lab"})
    m = ops.call("alchemy make", {"picture": "synth:12", "material": ["synth:7"],
                                  "method": "colour.lab", "size": 256})
    assert m.id.startswith("out:") and os.path.isfile(m.path)
    # Kept pictures go to output/edited/<family>/<date>/<picture>__<method>__<time>.png, and the
    # ledger records them, derived from the picture they came from, under the out: handle.
    folder, name = os.path.split(m.path)
    assert os.path.basename(os.path.dirname(folder)) == "colour"
    assert os.path.basename(os.path.dirname(os.path.dirname(folder))) == "edited"
    assert name.startswith("img_012__lab__")
    from arcana import ledger
    entry = ledger.get(m.id)
    assert entry["path"] == m.path and entry["op"] == "alchemy make" and entry["derived_from"] == "synth:12"
    from arcana.alchemy import inputs
    assert inputs.resolve(m.id) == m.path
    # With no size, a Keep has the picture's own size (the synthetic pictures are 64 x 48).
    full = ops.call("alchemy make", {"picture": "synth:12", "material": ["synth:7"], "method": "colour.lab"})
    assert full.size == [64, 48] and full.drawn is None
    again = ops.call("alchemy preview", {"picture": m.id, "material": ["synth:7"],
                                         "method": "pixels.displace", "size": 256})
    assert again.size[0] == 256


def test_the_tab_turns_its_controls_into_op_arguments():
    from arcana import ui_alchemy as ui
    v = {"method": "tokens.mix", "knob_ids": [{"key": "push"}, {"key": "stray"}], "knobs": [0.7, 1],
         "roles": {"T": "a.jpg", "1": "b.jpg", "2": None, "3": "c.jpg"}, "w1": 0.5, "w2": 1, "w3": 0.25,
         "lead": "picture", "zoom": 1, "crop": 0, "invert": ["on"],
         "fin_ids": [{"key": "exposure"}, {"key": "contrast"}], "fins": [0.3, 0.0], "scope": "mask",
         "layer1": "sky", "words1": "", "cov1": 0.5, "layer2": ui.WORDS, "words2": "woman", "cov2": 0.5,
         "comb2": "but not", "layer3": "none", "words3": "", "cov3": 0.5, "comb3": "and",
         "feather": 4, "grow": 0, "follow": 0, "ragged": 0, "amount": 1}
    a = ui.settings(v)
    assert a["knobs"] == {"push": 0.7}
    assert a["material"] == ["b.jpg", "c.jpg"] and a["weight"] == [0.5, 0.25]
    assert a["mask"] == ["sky", "prompt:woman"] and a["combine"] == ["but not"]
    assert "mask.invert=true" in a["set"]
    assert a["finish"] == {"exposure": 0.3, "scope": "mask"}
    from arcana.ops.alchemy import AlchemyPreviewArgs
    AlchemyPreviewArgs(**a)


def test_recipe_save_is_undone_like_any_write(tmp, ds, monkeypatch):
    monkeypatch.setattr(recipes, "_store", lambda: os.path.join(tmp, "recipes.json"))
    ops.exec_lines(['alchemy recipe save "warm sky" --method colour.lab --set finish.temperature=20'],
                   actor="cli")
    assert any(r.name == "warm sky" for r in recipes.own())
    ops.call("undo", {}, actor="cli")
    assert not any(r.name == "warm sky" for r in recipes.own())


def test_an_agent_names_pictures_by_handle(tmp, ds):
    path = ds[12]
    with pytest.raises(ops.Refusal, match="by handle"):
        ops.call("alchemy preview", {"picture": path, "material": ["synth:7"], "method": "colour.lab",
                                     "size": 256}, actor="agent")
    ops.call("alchemy preview", {"picture": "synth:12", "material": ["synth:7"], "method": "colour.lab",
                                 "size": 256}, actor="agent")
