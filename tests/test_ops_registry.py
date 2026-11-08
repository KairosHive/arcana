"""
The op registry: every op fully declared, every example run.

Each op's examples are complete command lines written for a reader. They are
run here against the synthetic dataset (with "japan" read as "synth"), so the
help an agent learns from cannot drift away from what the ops accept.

Run with:  python -m pytest tests/test_ops_registry.py -q
"""

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import core, ops  # noqa: E402
from arcana.ops import registry  # noqa: E402
from pydantic import Field  # noqa: E402

import synth  # noqa: E402


# The slow examples use this machine's ModFlows checkpoint, found here before
# a fixture points the data dir at a scratch folder that has none.
REAL_CHECKPOINT = None
if os.environ.get("ARCANA_SLOW_EXAMPLES"):
    from arcana import color_transfer as _ct
    REAL_CHECKPOINT = _ct.checkpoint_path()


@pytest.fixture
def ds(tmp, monkeypatch):
    if REAL_CHECKPOINT is not None:
        from arcana import color_transfer as _ct
        monkeypatch.setattr(_ct, "checkpoint_path", lambda: REAL_CHECKPOINT)
    monkeypatch.setenv("ARCANA_NO_BROWSER", "1")
    out = synth.use(tmp, monkeypatch, palette_features=True)
    from arcana import style_transfer
    monkeypatch.setattr(style_transfer, "describe", lambda *a, **k: "lake, mist")
    monkeypatch.setitem(registry.FLAG_PROBES, "gpu", lambda: False)
    return out


@pytest.fixture
def scratch():
    """Ops registered by a test, removed after it."""
    names = []
    yield names
    for n in names:
        registry.unregister(n)


def _as_synth(line: str) -> str:
    return re.sub(r"\bjapan\b", "synth", line)


def _files(root):
    return {os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs}


# ─────────────────────────── declarations ───────────────────────────

def test_every_op_is_fully_declared():
    seen = set()
    for o in ops.all_ops():
        assert o.name not in seen
        seen.add(o.name)
        assert registry.NAME_RE.match(o.name), o.name
        assert o.kind in registry.KINDS
        assert o.doc.strip() and len(o.summary) < 90, (o.name, o.summary)
        assert issubclass(o.args, ops.Args) and o.args.model_config["extra"] == "forbid"
        assert issubclass(o.result, ops.Result)
        assert o.examples, f"{o.name} has no example"
        for name, info in o.args.model_fields.items():
            assert info.description, f"{o.name}: field {name} has no description"


VOCABULARY = {
    # P0: reads
    "orient", "op list", "help", "settings show", "dataset list", "dataset show",
    "dataset survey", "model list", "search text", "search facets", "search like",
    "search palette", "item show", "item sheet", "item palette", "item describe",
    # P1: state and boards
    "settings set", "undo", "redo", "history", "board list", "board show", "board create",
    "board copy", "board save", "board add", "board remove", "board move", "board note",
    "board set", "board section add", "board section set", "board section remove",
    "board palette set", "board palette extract", "board export", "board delete",
    "item export",
    # P2: colour, style, generation, jobs
    "colour methods", "colour transfer", "colour lut", "colour harmonize", "style methods",
    "style transfer", "style control-map", "generate", "model download",
    "job list", "job show", "job wait", "job cancel",
    # P3: boards as pages
    "board render", "board open",
    # P4: the dataset lifecycle
    "dataset create", "dataset extend", "dataset finish", "dataset marks", "dataset relocate",
    "dataset migrate", "dataset delete",
}

# Examples that load a large model. They run when ARCANA_SLOW_EXAMPLES is set,
# and their behaviour is tested with stubs in test_colour_ops.py either way.
# `model download` fetches hundreds of megabytes, so its example never runs here.
SLOW = {"generate", "colour lut", "alchemy make"}
NETWORK = {"model download"}
# Examples that index or rewrite a real folder. test_lifecycle.py runs these
# ops against the synthetic dataset with the encoding stubbed.
INDEXING = {"dataset create", "dataset extend", "dataset finish", "dataset relocate",
            "dataset migrate"}


def test_the_vocabulary_is_registered():
    names = {o.name for o in ops.all_ops()}
    assert VOCABULARY <= names, sorted(VOCABULARY - names)


def test_the_person_keeps_what_only_the_person_decides():
    for name in ("settings set", "board delete"):
        assert "agent" not in ops.get(name).actors, name


def test_help_renders_and_names_every_field():
    for o in ops.all_ops():
        text = ops.help_text(o)
        assert text.startswith(o.name)
        for name, info in o.args.model_fields.items():
            spelled = f"<{name}>" if (info.json_schema_extra or {}).get("positional") is not None \
                else "--" + name.replace("_", "-")
            assert spelled in text, (o.name, spelled)
        for e in o.examples:
            assert e in text


def test_the_schema_generates_and_forbids_unknown_fields():
    schema = ops.grammar.schema()
    json.dumps(schema)
    assert set(schema) == {o.name for o in ops.all_ops()}
    for name, entry in schema.items():
        assert entry["args"].get("additionalProperties") is False, name


# ─────────────────────────── examples run ───────────────────────────

def _fixture_board(tmp, author):
    """The board the board ops' examples name: monsoon, with sections s1 and s2."""
    from arcana import boards
    folder = boards.boards_dir()
    for name in os.listdir(folder):
        os.remove(os.path.join(folder, name))
    idx2path = core.open_dataset("synth").idx2path
    keys = [12, 7, 3, 1, 2]
    rec = {"schema": 2, "name": "Monsoon", "slug": "monsoon", "author": author,
           "items": [idx2path[k] for k in keys],
           "details": {idx2path[k]: {"handle": f"synth:{k}",
                                     "section": "s1" if k in (12, 7) else "s2"} for k in keys},
           "sections": [{"id": "s1", "title": "Water"}, {"id": "s2", "title": "Shelter"}]}
    with open(os.path.join(folder, "monsoon.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)


@pytest.mark.parametrize("actor,pixels_on", [("cli", False), ("agent", False), ("agent", True)])
def test_every_example_runs(tmp, ds, actor, pixels_on):
    """Each example from scratch: reads change nothing, writes are allowed to."""
    from arcana import jobs
    synth.pixels(tmp, pixels_on)
    ctx = ops.context(actor)
    slow = bool(os.environ.get("ARCANA_SLOW_EXAMPLES"))
    for o in ops.all_ops():
        if actor not in o.actors or o.name in NETWORK | INDEXING or (o.name in SLOW and not slow):
            continue
        for line in o.examples:
            _fixture_board(tmp, "agent" if actor == "agent" else "person")
            if o.name == "board create":
                os.remove(os.path.join(tmp, "boards", "monsoon.json"))
            if o.name.startswith("job "):
                # A job that waits a moment, so cancel finds it running.
                jid = jobs.MANAGER.submit(lambda h: __import__("time").sleep(0.3) or {"text": "ok"},
                                          kind="test", label="example")
                line = line.replace("3f9c1e2a7b4d", jid)
            before = _files(tmp)
            batch = ops.exec_lines([_as_synth(line)], ctx=ctx)
            step = batch.steps[0]
            assert step.refusal is None, (line, step.refusal and step.refusal["reason"])
            assert step.result.text().strip()
            json.dumps(ops.render.result_json(step.result))
            if o.kind == "read":
                assert _files(tmp) == before, line


# ─────────────────────────── acceptance ───────────────────────────

@pytest.mark.parametrize("phrase", ["water", "forest", "spaceship"])
def test_search_text_returns_the_apps_prompt_search_hits(tmp, ds, phrase):
    """The app's Prompt Search ranks with core.search and folds twins with cosine_group."""
    index, idx2path = core.load_index("synth")
    app = core.search(index, idx2path, phrase, 12)
    leaders = [g["keys"][0] for g in core.cosine_group([k for k, _, _ in app],
                                                       [p for _, p, _ in app], index, 0.08)]
    r = ops.call("search text", {"dataset": "synth", "phrase": [phrase], "k": 12})
    mine = [int(h.handle.split(":")[1]) for h in r.facets[0].hits]
    assert mine[:len(leaders)] == leaders


# ─────────────────────────── refusals ───────────────────────────

def test_an_unknown_op_suggests_one():
    with pytest.raises(ops.Refusal, match="Did you mean `search text`"):
        ops.call("search txt")


def test_bad_arguments_are_refused_in_the_callers_terms(tmp, ds):
    with pytest.raises(ops.Refusal) as e:
        ops.call("search text", {"dataset": "synth", "phrase": ["water"], "k": 200})
    assert "--k must be between 1 and 48, got 200" in e.value.reason
    assert e.value.help.startswith("search text")

    with pytest.raises(ops.Refusal, match="--phrase is required"):
        ops.call("search text", {"dataset": "synth"})
    with pytest.raises(ops.Refusal, match="--mode must be one of"):
        ops.call("search like", {"handles": ["synth:1"], "mode": "vibes"})
    with pytest.raises(ops.Refusal, match="takes 1 to 6 values, got 7"):
        ops.call("search text", {"dataset": "synth", "phrase": ["w"] * 7})
    with pytest.raises(ops.Refusal, match="not an option of `search text`"):
        ops.call("search text", {"dataset": "synth", "phrase": ["w"], "kk": 3})


def test_a_failure_inside_an_op_comes_back_as_a_refusal_with_help(tmp, ds):
    with pytest.raises(ops.Refusal) as e:
        ops.call("search text", {"dataset": "nope", "phrase": ["water"]})
    assert "Known datasets: synth" in e.value.reason
    assert e.value.op == "search text" and "Examples" in e.value.help

    with pytest.raises(ops.Refusal, match="Not in synth: synth:999"):
        ops.call("search like", {"handles": ["synth:999"]})
    with pytest.raises(ops.Refusal, match="one dataset at a time"):
        ops.call("search like", {"handles": ["synth:1", "other:2"]})


def test_actors_and_capabilities_gate_ops(tmp, ds, scratch, monkeypatch):
    class NoArgs(ops.Args):
        pass

    ops.register(ops.Op("test personal", "read", NoArgs, ops.TextResult,
                        lambda a, c: ops.TextResult(body="ok"), doc="For the person.",
                        examples=("test personal",), actors=frozenset({"person", "cli"})))
    scratch.append("test personal")
    ops.register(ops.Op("test painter", "effect", NoArgs, ops.TextResult,
                        lambda a, c: ops.TextResult(body="ok"), doc="Needs diffusers.",
                        examples=("test painter",), needs=frozenset({"diffusers"})))
    scratch.append("test painter")
    monkeypatch.setitem(registry.FLAG_PROBES, "diffusers", lambda: False)

    assert ops.call("test personal", actor="cli").text() == "ok"
    with pytest.raises(ops.Refusal, match="is for the person"):
        ops.call("test personal", actor="agent")
    with pytest.raises(ops.Refusal, match="needs the diffusers library"):
        ops.call("test painter")
    listed = ops.call("op list", {"noun": "test"}, actor="agent").text()
    assert "test painter" in listed and "(needs diffusers)" in listed
    assert "(for the person)" in listed


def test_register_refuses_what_would_break_the_rule(scratch):
    from pydantic import BaseModel

    class Loose(BaseModel):
        pass

    class NoArgs(ops.Args):
        pass

    class Needy(ops.Args):
        n: int = Field(description="A number.")

    def body(a, c):
        return ops.TextResult(body="")

    def bad(**kw):
        base = dict(name="test thing", kind="read", args=NoArgs, result=ops.TextResult,
                    run=body, doc="Does a thing.", examples=("test thing",))
        base.update(kw)
        return ops.Op(**base)

    with pytest.raises(TypeError, match="already exists"):
        ops.register(bad(name="search text"))
    with pytest.raises(TypeError, match="needs a doc"):
        ops.register(bad(doc=" "))
    with pytest.raises(TypeError, match="subclass ops.Args"):
        ops.register(bad(args=Loose))
    with pytest.raises(TypeError, match="lowercase words"):
        ops.register(bad(name="Thing"))
    with pytest.raises(TypeError, match="does not parse"):
        ops.register(bad(args=Needy, examples=("test thing --n ten",)))
    assert ops.get("test thing") is None


# ─────────────────────────── batches ───────────────────────────

def test_a_batch_stops_at_its_first_refusal(tmp, ds):
    batch = ops.exec_lines(["dataset list", "search text nope --phrase water",
                            "dataset survey synth", "", "# a comment"])
    assert [s.line for s in batch.steps] == ["dataset list", "search text nope --phrase water",
                                            "dataset survey synth"]
    first, refused, after = batch.steps
    assert first.result is not None
    assert "Known datasets" in refused.refusal["reason"]
    assert after.result is None and after.refusal is None
    assert not batch.ok and batch.refused is refused
    text = ops.render.batch_text(batch)
    assert "Line 2 was refused" in text and "(not run)" in text
    assert ops.render.batch_json(batch)["steps"][2]["ran"] is False


def test_a_help_line_answers_without_running(tmp, ds):
    step = ops.exec_lines(["search text --help"]).steps[0]
    assert step.help and step.result.text().startswith("search text <dataset>")


# ─────────────────────────── pixels ───────────────────────────

def test_a_blind_agent_gets_no_pictures_and_is_told(tmp, ds):
    synth.pixels(tmp, False)
    blind = ops.call("item sheet", {"handles": ["synth:1", "synth:2"]}, actor="agent")
    assert not blind.attachments and "stays on this machine" in blind.text()
    search = ops.call("search text", {"dataset": "synth", "phrase": ["water"]}, actor="agent")
    assert not search.attachments
    assert "#" in search.facets[0].hits[0].palette[0]      # palettes stand in for pictures

    synth.pixels(tmp, True)
    seen = ops.call("item sheet", {"handles": ["synth:1", "synth:2"]}, actor="agent")
    assert len(seen.attachments) == 1 and seen.attachments[0].data[:2] == b"\xff\xd8"


def test_the_renderer_withholds_pictures_whatever_an_op_attached(tmp, ds):
    synth.pixels(tmp, False)
    r = ops.TextResult(body="x", attachments=[ops.Attachment(data=b"\xff\xd8")])
    assert ops.render.visible_attachments(r, ops.context("agent")) == []
    assert len(ops.render.visible_attachments(r, ops.context("cli"))) == 1


def test_attachments_never_reach_json():
    r = ops.TextResult(body="x", attachments=[ops.Attachment(caption="c", data=b"\xff\xd8")])
    dumped = ops.render.result_json(r)
    assert dumped["attachments"] == [{"caption": "c", "mime": "image/jpeg", "path": None}]
