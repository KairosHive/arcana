"""
The text grammar: a line into an op and its arguments, and back.

A scratch op with one field of every shape the grammar knows is registered for
these tests, so each rule in grammar.py's header is checked on its own.

Run with:  python -m pytest tests/test_grammar.py -q
"""

import json
import os
import sys
from typing import Literal

import pytest
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import ops  # noqa: E402
from arcana.ops import grammar, registry  # noqa: E402


class Sub(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    phrases: list[str] = Field(default_factory=list)


class EchoArgs(ops.Args):
    target: str = ops.positional(0, description="What to echo.")
    rest: list[str] = ops.positional(1, [], description="More words.")
    k: int = Field(12, ge=1, le=48, description="A count.")
    ratio: float = Field(0.5, ge=-1.0, le=1.0, description="A ratio.")
    loud: bool = Field(False, description="Shout.")
    describe: bool | None = Field(None, description="Maybe.")
    tag: list[str] = Field(default_factory=list, max_length=3, description="Tags.")
    mode: Literal["semantic", "palette"] = Field("semantic", description="A mode.")
    facet: list[Sub] = Field(default_factory=list, description="Nested ideas.")
    one: Sub | None = Field(None, description="One nested idea.")
    path: str | None = Field(None, description="A path.")


def _echo(a, ctx):
    return ops.TextResult(body=a.model_dump_json())


@pytest.fixture(autouse=True)
def echo_op():
    ops.register(ops.Op("test echo", "read", EchoArgs, ops.TextResult, _echo,
                        doc="Echo the arguments back.", examples=("test echo x",)))
    yield
    registry.unregister("test echo")


def parse(line):
    p = ops.parse_line(line)
    assert p.op.name == "test echo"
    return p.args


def run(line):
    return json.loads(ops.call("test echo", parse(line)).text())


# ─────────────────────────── splitting ───────────────────────────

def test_windows_paths_survive_quoted_or_not():
    assert grammar.split(r"x C:\Users\me\Pictures") == ["x", r"C:\Users\me\Pictures"]
    assert grammar.split(r'x "D:\My Photos\2024"') == ["x", r"D:\My Photos\2024"]
    assert grammar.split("x C:/Users/me/Pictures") == ["x", "C:/Users/me/Pictures"]


def test_hash_is_a_colour_not_a_comment():
    assert grammar.split("search palette japan --colour #1f2a36 --colour '#b5651d:0.4'") == \
        ["search", "palette", "japan", "--colour", "#1f2a36", "--colour", "#b5651d:0.4"]


def test_json_in_single_quotes_arrives_whole():
    words = grammar.split("""x --facet '{"name": "w", "phrases": ["rain at night"]}'""")
    assert json.loads(words[2]) == {"name": "w", "phrases": ["rain at night"]}


def test_an_unclosed_quote_is_refused():
    with pytest.raises(ops.Refusal, match="closing quotation"):
        grammar.split('x "never closed')


@pytest.mark.parametrize("tokens", [
    ["a", "b c"], ["it's"], ['say "hi"'], ["both ' and \""], [r"C:\a b\c"], ["#fff:0.2"], [""],
])
def test_join_is_the_inverse_of_split(tokens):
    assert grammar.split(grammar.join(tokens)) == tokens


# ─────────────────────────── field shapes ───────────────────────────

def test_scalars_coerce_and_take_either_spelling():
    got = run("test echo x --k 3 --ratio=-0.25 --path C:\\tmp\\out")
    assert got["k"] == 3 and got["ratio"] == -0.25 and got["path"] == "C:\\tmp\\out"


def test_bools_take_no_value_and_have_a_negative():
    assert run("test echo x --loud")["loud"] is True
    assert run("test echo x --no-describe")["describe"] is False
    assert run("test echo x --loud=off")["loud"] is False
    assert run("test echo x")["describe"] is None


def test_lists_repeat_the_flag():
    assert run("test echo x --tag a --tag 'b c'")["tag"] == ["a", "b c"]


def test_positionals_fill_in_order_and_a_trailing_list_takes_the_rest():
    got = run("test echo first second third --k 2")
    assert got["target"] == "first" and got["rest"] == ["second", "third"]
    assert run("test echo -- --odd")["target"] == "--odd"


def test_nested_models_are_json_one_or_many_per_flag():
    got = run("""test echo x --facet '{"name": "a"}' --facet '[{"name": "b"}, {"name": "c"}]'""")
    assert [f["name"] for f in got["facet"]] == ["a", "b", "c"]
    assert run("""test echo x --one '{"name": "solo", "phrases": ["p"]}'""")["one"]["phrases"] == ["p"]


def test_the_whole_model_can_come_as_json_or_a_file(tmp):
    assert run("""test echo --args '{"target": "t", "k": 5}'""")["k"] == 5
    req = os.path.join(tmp, "req.json")
    with open(req, "w", encoding="utf-8") as fh:
        json.dump({"target": "from file", "mode": "palette"}, fh)
    got = run(f"test echo --args @{req}")
    assert got["target"] == "from file" and got["mode"] == "palette"


# ─────────────────────────── refusals ───────────────────────────

@pytest.mark.parametrize("line,match", [
    ("test echo x --kk 3", "`--kk` is not an option of `test echo`. Did you mean `--k`?"),
    ("test echo x --k", "`--k` needs a value"),
    ("test echo x --k 1 --k 2", "given twice"),
    ("test echo x --loud=maybe", "on or off"),
    ("test echo x --facet '{nope'", "takes JSON"),
    ("""test echo x --args '{"target": "y"}'""", "both in --args and on the line"),
    ("test echo --target a b", "both bare and as `--target`"),
])
def test_line_mistakes_are_refused_with_help(line, match):
    with pytest.raises(ops.Refusal, match=match) as e:
        ops.parse_line(line)
    assert e.value.help.startswith("test echo")


@pytest.mark.parametrize("line,match", [
    ("test echo x --k 99", "--k must be between 1 and 48, got 99"),
    ("test echo x --k ten", "--k must be a whole number"),
    ("test echo x --mode vibes", "--mode must be one of"),
    ("test echo x --tag a --tag b --tag c --tag d", "--tag takes at most 3 values"),
    ("test echo", "<target> is required"),
])
def test_bad_values_are_refused_by_the_model_in_line_terms(line, match):
    with pytest.raises(ops.Refusal, match=match):
        ops.call("test echo", parse(line))


def test_unknown_ops_get_the_nearest_names():
    with pytest.raises(ops.Refusal, match="No op called `serch`"):
        ops.parse_line("serch text japan")
    with pytest.raises(ops.Refusal, match="The `search` ops are"):
        ops.parse_line("search txt japan")


def test_the_longest_registered_prefix_names_the_op():
    assert ops.parse_line("help search text").op.name == "help"
    assert ops.parse_line("search text japan --phrase x").op.name == "search text"


def test_help_flags_ask_for_help():
    assert ops.parse_line("test echo --help").help
    assert ops.parse_line("test echo x -h").help


# ─────────────────────────── help ───────────────────────────

def test_help_names_every_flag_with_its_bounds_and_default():
    text = ops.help_text(ops.get("test echo"))
    for term in ("<target>", "<rest>...", "--k N", "--loud, --no-loud", "--mode semantic|palette",
                 "--facet JSON", "1 to 48, default 12", "repeatable", "Examples"):
        assert term in text, term


def test_usage_wraps_between_terms_never_inside_one():
    usage = ops.help_text(ops.get("search text")).split("\n(")[0].split("\n")
    assert len(usage) > 1
    for line in usage[1:]:
        assert line.startswith("    [") or line.startswith("    --"), line


# ─────────────────────────── completion ───────────────────────────

def test_completion_walks_op_names_flags_and_choices():
    assert grammar.complete("se") == ["search", "sequence", "settings"]
    assert grammar.complete("search ") == ["across", "facets", "like", "palette", "text"]
    assert grammar.complete("arcana test echo x --lo") == ["--loud"]
    assert "--no-loud" in grammar.complete("test echo x --no")
    assert grammar.complete("test echo x --mode ") == ["semantic", "palette"]
    assert grammar.complete("test echo x --mode p") == ["palette"]


def test_completion_offers_datasets_for_a_dataset_argument(tmp, monkeypatch):
    import synth
    synth.use(tmp, monkeypatch)
    assert grammar.complete("dataset survey ") == ["synth"]
    assert grammar.complete("search text sy") == ["synth"]


def test_completion_scripts_exist_for_three_shells():
    assert "Register-ArgumentCompleter" in grammar.completion_script("powershell")
    assert "complete -o default" in grammar.completion_script("bash")
    assert "bashcompinit" in grammar.completion_script("zsh")
    with pytest.raises(ops.Refusal):
        grammar.completion_script("fish")
