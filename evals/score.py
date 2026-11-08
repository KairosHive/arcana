"""
Score moodboards built for the eval briefs.

A run builds one board per brief and variant, with the id
eval-<brief id>-<variant> (for example eval-monsoon-skills and
eval-monsoon-plain). This reads each through the ops and prints one row per
board with what can be measured without a person:

    sections      angles the board covers, as sections with at least two pictures
    reasons       sections that state why they are there
    heroes        1 is right
    near_dups     pairs of near-duplicates, 0 is right
    diversity     mean distance between pictures
    palette       whether the board has a palette
    derived       share of derived or generated pictures, at most a third
    rendered      whether an HTML board exists in output/boards

The person's ranking of each pair is the measure that decides. These numbers
say whether a board meets the floor the moodboard skill sets.

Run with:  python evals/score.py [--json]
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from arcana import boards, ops, paths  # noqa: E402


def score(slug: str) -> dict:
    shown = ops.call("board show", {"board": slug}, actor="cli")
    m = shown.metrics
    covered = [s for s in shown.sections if len(s.items) >= 2]
    html = os.path.join(paths.subdir("output"), "boards", f"{slug}.html")
    return {
        "board": slug,
        "pictures": m.count,
        "sections": len(covered),
        "reasons": sum(1 for s in shown.sections if s.rationale.strip()),
        "heroes": len(m.heroes),
        "near_dups": len(m.near_duplicates),
        "diversity": round(m.diversity, 3) if m.diversity is not None else None,
        "palette": bool(shown.palette),
        "derived": round((m.derived + m.generated) / m.count, 3) if m.count else 0.0,
        "rendered": os.path.exists(html),
        "meets_floor": (len(covered) >= 4 and len(m.heroes) == 1 and not m.near_duplicates
                        and bool(shown.palette) and m.count > 0
                        and (m.derived + m.generated) / m.count <= 1 / 3
                        and all(s.rationale.strip() for s in shown.sections)),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(os.path.join(HERE, "briefs.json"), encoding="utf-8") as fh:
        briefs = json.load(fh)["briefs"]
    have = {b["slug"] for b in boards.listing()}
    rows = []
    for b in briefs:
        for variant in ("skills", "plain"):
            slug = f"eval-{b['id']}-{variant}"
            if slug in have:
                rows.append({"brief": b["id"], "variant": variant, **score(slug)})
    if "--json" in argv:
        print(json.dumps(rows, indent=1))
        return 0
    if not rows:
        print("No eval boards yet. Build eval-<brief>-skills and eval-<brief>-plain first "
              "(evals/README.md).")
        return 0
    cols = ["brief", "variant", "pictures", "sections", "reasons", "heroes", "near_dups",
            "diversity", "palette", "derived", "rendered", "meets_floor"]
    print("  ".join(f"{c:>11}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>11}" for c in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
