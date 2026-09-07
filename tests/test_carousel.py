"""
Twin carousels: two panels, one page, and no crosstalk between them.

Prompt search and the moodboard's Find similar both render carousels into the
same results panel. They used to mount components under the same pattern id and
share one set of stores, which produced three distinct faults:

  * the moodboard's arrows moved nothing;
  * an arrow in prompt search repainted its cards with the moodboard's images;
  * pressing Search after a Find similar put the moodboard's results straight
    back, because removing the moodboard's step-to buttons fires its own
    ALL-pattern Input and n_clicks was still 1 from the last real press.

And one that showed in both: a newly mounted arrow button fires the nav
callback with n_clicks=0, which was read as a click, so every card opened on
its LAST twin -- "15/15" before anyone had touched it.

Run with:  python -m pytest tests/test_carousel.py -q
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcana import arcana as app  # noqa: E402


def _groups(n_twins=3, gid="g0"):
    return [{"gid": gid,
             "keys": list(range(n_twins)),
             "paths": [f"C:/img/{gid}_{i}.jpg" for i in range(n_twins)],
             "marks": [["", "", ""] for _ in range(n_twins)]}]


def _fake_ctx(monkeypatch, triggered_id, value):
    """Stand in for dash.ctx as it looks inside a pattern-matching callback."""
    fake = types.SimpleNamespace(
        triggered_id=triggered_id,
        triggered=[{"prop_id": "x.n_clicks", "value": value}],
    )
    monkeypatch.setattr(app, "ctx", fake)


def _drive(monkeypatch, triggered_id, value, groups, car_state):
    _fake_ctx(monkeypatch, triggered_id, value)
    order = [g["gid"] for g in groups if len(g["paths"]) > 1]
    return app.nav_carousel(
        left_clicks=[0] * len(order), right_clicks=[0] * len(order),
        groups=groups, order=order, car_state=car_state, spec_on=False)


# ───────────────────── the phantom click on mount ─────────────────────
def test_mounting_a_card_does_not_count_as_a_click(monkeypatch):
    """
    Dash fires the callback when new arrow buttons appear, with n_clicks=0.

    Read as a left click that ran (0 - 1) % 15 = 14, so a fifteen-twin card
    opened at "15/15". Nothing had been clicked.
    """
    groups = _groups(15)
    state, srcs, _sets, counters, _marks, _audio = _drive(
        monkeypatch, {"type": "left", "owner": "search", "gid": "g0"}, 0,
        groups, {"g0": 0})
    assert state["g0"] == 0
    assert counters[0] == "1/15"
    assert "g0_0.jpg" in srcs[0]


def test_a_real_click_still_steps(monkeypatch):
    groups = _groups(3)
    state, srcs, _s, counters, _m, _a = _drive(
        monkeypatch, {"type": "right", "owner": "search", "gid": "g0"}, 1,
        groups, {"g0": 0})
    assert state["g0"] == 1
    assert counters[0] == "2/3"
    assert "g0_1.jpg" in srcs[0]


def test_left_from_the_first_frame_wraps_to_the_last(monkeypatch):
    """The wrap itself is correct and wanted -- it was only the phantom that hurt."""
    groups = _groups(15)
    state, _s, _ss, counters, _m, _a = _drive(
        monkeypatch, {"type": "left", "owner": "search", "gid": "g0"}, 1,
        groups, {"g0": 0})
    assert state["g0"] == 14
    assert counters[0] == "15/15"


def test_a_store_refresh_moves_nothing(monkeypatch):
    """New results arriving is not a navigation event."""
    groups = _groups(4)
    state, _s, _ss, counters, _m, _a = _drive(
        monkeypatch, "grouped-results", None, groups, {"g0": 0})
    assert state["g0"] == 0
    assert counters[0] == "1/4"


def test_only_the_clicked_group_moves(monkeypatch):
    groups = _groups(3, "g0") + _groups(5, "g1")
    state, _s, _ss, counters, _m, _a = _drive(
        monkeypatch, {"type": "right", "owner": "search", "gid": "g1"}, 1,
        groups, {"g0": 0, "g1": 0})
    assert state == {"g0": 0, "g1": 1}
    assert counters == ["1/3", "2/5"]


def test_marks_follow_the_frame(monkeypatch):
    """The badge names the picture on screen, not the first of the group."""
    groups = _groups(2)
    groups[0]["marks"] = [["★★★☆☆", "Red", "01 Jan 2025"],
                          ["★★★★★", "Yellow", "02 Jan 2025"]]
    _st, _s, _ss, _c, marks, _a = _drive(
        monkeypatch, {"type": "right", "owner": "search", "gid": "g0"}, 1,
        groups, {"g0": 0})
    rendered = str(marks[0])
    assert "Yellow" in rendered and "02 Jan 2025" in rendered


# ───────────────────── the two panels are separate ─────────────────────
def test_each_panel_has_its_own_stores():
    """
    Sharing them is what let one panel's results drive the other's carousels.
    """
    assert set(app.CAROUSEL_OWNERS) == {"search", "moodboard"}
    stores = [s for triple in app.CAROUSEL_OWNERS.values() for s in triple]
    assert len(stores) == len(set(stores)), f"panels share a store: {stores}"


def test_every_carousel_component_declares_its_owner():
    """
    An id without an owner is matched by both panels' callbacks at once, which
    is the collision this whole module is about.
    """
    import pathlib
    source = (pathlib.Path(app.__file__)).read_text(encoding="utf-8")
    kinds = ("carousel-img", "carousel-counter", "carousel-marks",
             "carousel-audio")
    unowned = []
    for kind in kinds:
        needle = '{"type": "%s", "gid":' % kind          # no owner between them
        if needle in source:
            unowned.append(kind)
    assert not unowned, f"carousel ids with no owner: {unowned}"


def test_both_owners_are_registered():
    """A panel with ids but no callback has arrows that do nothing at all."""
    import pathlib
    source = (pathlib.Path(app.__file__)).read_text(encoding="utf-8")
    for owner in app.CAROUSEL_OWNERS:
        assert '"owner": "%s"' % owner in source, f"{owner} mounts no carousels"


def main():
    print("This suite uses pytest's monkeypatch; run it with:")
    print("    python -m pytest tests/test_carousel.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
