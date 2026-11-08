"""
The plugin: its skills, manifest, MCP config, marketplace and eval set.

Every op line in a skill is one an agent will send, so each must parse and
validate against the current vocabulary. A skill that drifts from the ops
teaches a model to make mistakes.

Run with:  python -m pytest tests/test_skills.py -q
"""

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from arcana import ops  # noqa: E402
from arcana.ops import registry  # noqa: E402

PLUGIN = os.path.join(ROOT, "plugin")
SKILLS = ("moodboard", "board-design", "harmonize")


def _skill(name):
    with open(os.path.join(PLUGIN, "skills", name, "SKILL.md"), encoding="utf-8") as fh:
        return fh.read()


def _op_lines(text):
    for block in re.findall(r"```ops\n(.*?)```", text, re.S):
        for line in block.splitlines():
            if line.strip():
                yield line.strip()


@pytest.mark.parametrize("name", SKILLS)
def test_each_skill_has_front_matter(name):
    text = _skill(name)
    head = text.split("---")[1]
    assert re.search(rf"^name: {name}$", head, re.M)
    assert re.search(r"^description: .{80,}", head, re.M)


@pytest.mark.parametrize("name", SKILLS)
def test_every_op_line_in_a_skill_parses_and_validates(name):
    lines = list(_op_lines(_skill(name)))
    assert lines, f"{name} teaches no op lines"
    for line in lines:
        parsed = ops.parse_line(line)
        registry.validate(parsed.op, parsed.args)


def test_the_moodboard_skill_walks_the_whole_workflow():
    used = {ops.parse_line(l).op.name for l in _op_lines(_skill("moodboard"))}
    assert {"dataset survey", "search facets", "board create", "board section add", "board add",
            "board show", "board palette extract", "colour harmonize", "generate",
            "board render"} <= used


def test_the_plugin_manifest_mcp_config_and_marketplace():
    manifest = json.load(open(os.path.join(PLUGIN, ".claude-plugin", "plugin.json"), encoding="utf-8"))
    assert manifest["name"] == "arcana" and manifest["description"]
    server = json.load(open(os.path.join(PLUGIN, ".mcp.json"), encoding="utf-8"))["mcpServers"]["arcana"]
    assert "arcana-mcp" in server["args"] and "${CLAUDE_PLUGIN_ROOT}/.." in server["args"]
    market = json.load(open(os.path.join(ROOT, ".claude-plugin", "marketplace.json"), encoding="utf-8"))
    src = market["plugins"][0]["source"]
    assert os.path.isdir(os.path.join(ROOT, src)) and market["plugins"][0]["name"] == "arcana"


def test_the_moodboard_prompt_carries_the_skill():
    pytest.importorskip("mcp")
    from arcana import mcp_server
    text = mcp_server.moodboard("erosion", "abstract")
    assert "Gather wide" in text and "dataset survey abstract" in text
    assert not text.lstrip().startswith("---")


def test_the_eval_set_is_five_briefs_on_two_datasets():
    briefs = json.load(open(os.path.join(ROOT, "evals", "briefs.json"), encoding="utf-8"))["briefs"]
    assert len(briefs) == 5
    assert {b["dataset"] for b in briefs} == {"japan", "abstract"}
    assert all(len(b["angles"]) >= 4 for b in briefs)


def test_the_scorer_reads_an_eval_board(tmp, monkeypatch):
    sys.path.insert(0, HERE)
    import synth
    synth.use(tmp, monkeypatch)
    sys.path.insert(0, os.path.join(ROOT, "evals"))
    import score
    b = ops.exec_lines([
        'board create "E" --id eval-monsoon-skills',
        *[f'board section add eval-monsoon-skills --id s{i} --title T{i} --rationale "why {i}"'
          for i in range(1, 5)],
        "board add eval-monsoon-skills synth:0 --section s1 --role hero",
        "board add eval-monsoon-skills synth:5 --section s1",
        "board add eval-monsoon-skills synth:10 synth:20 --section s2",
        "board add eval-monsoon-skills synth:21 synth:22 --section s3",
        "board add eval-monsoon-skills synth:23 synth:24 --section s4",
        "board palette set eval-monsoon-skills --colour '#112233'",
    ])
    assert b.ok, b.refused and b.refused.refusal
    row = score.score("eval-monsoon-skills")
    assert row["sections"] == 4 and row["heroes"] == 1 and row["meets_floor"]
