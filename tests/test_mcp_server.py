"""
The MCP server: one tool, `exec`, taking op lines.

The tool is called as a plain function first, then over a real stdio pipe,
because the failure that matters most there -- a stray print() corrupting the
protocol -- only shows up over a pipe. One test starts the ops routes on a
local port and checks the tool forwards to them, the way it does when the app
is open. The ops' own behaviour is tested in test_read_ops.py and
test_exec.py.

Run with:  python -m pytest tests/test_mcp_server.py
"""

import json
import os
import sys
import threading

import pytest

pytest.importorskip("mcp")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from arcana import mcp_server as srv, server  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def ds(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_NO_SERVER", "1")      # never reach a real app on this machine
    return synth.use(tmp, monkeypatch)


def texts(out):
    return "\n".join(c.text for c in out if c.type == "text")


def images(out):
    return [c for c in out if c.type == "image"]


def test_there_is_one_tool_plus_resources_and_a_prompt():
    import anyio
    tools = anyio.run(srv.mcp.list_tools)
    assert [t.name for t in tools] == ["exec"]
    assert "commands" in tools[0].input_schema["properties"]
    assert "orient" in tools[0].description
    resources = {str(r.uri) for r in anyio.run(srv.mcp.list_resources)}
    assert {"arcana://orient", "arcana://ops"} <= resources
    assert [p.name for p in anyio.run(srv.mcp.list_prompts)] == ["moodboard"]


def test_a_batch_comes_back_as_text_with_the_mode(tmp, ds):
    out = srv.run(["dataset list", "search text synth --phrase water --k 3"])
    t = texts(out)
    assert "[1] dataset list" in t and "## water — strong" in t
    assert "Mode: blind" in t and not images(out)


def test_pictures_reach_the_agent_only_with_pixels_on(tmp, ds):
    synth.pixels(tmp, True)
    out = srv.run(["search text synth --phrase water --k 3"])
    assert len(images(out)) == 1 and "Mode: pixels" in texts(out)


def test_a_refusal_is_an_error_the_model_can_read(tmp, ds):
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError) as e:
        srv.run(["search text nope --phrase water"])
    assert "Known datasets: synth" in str(e.value)
    assert "search text <dataset>" in str(e.value)        # the op's help comes with it


def test_the_agent_cannot_act_as_the_person(tmp, ds):
    from mcp.server.mcpserver.exceptions import ToolError
    srv.run(['board create "Mine" --id mine'])
    with pytest.raises(ToolError, match="is for the person"):
        srv.run(["board delete mine"])
    with pytest.raises(ToolError, match="is for the person"):
        srv.run(["settings set --pixels on"])


def test_a_refused_batch_leaves_nothing_behind(tmp, ds):
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError, match="undone"):
        srv.run(['board create "Half" --id half', "board add half synth:0", "board add half synth:999"])
    assert not os.path.exists(os.path.join(tmp, "boards", "half.json"))


def test_the_moodboard_prompt_names_the_theme_and_dataset():
    text = srv.moodboard("monsoon melancholy", "japan")
    assert "monsoon melancholy" in text and "dataset survey japan" in text


# ─────────────────────────── forwarding ───────────────────────────

def test_with_the_app_open_the_tool_forwards_to_it(tmp, ds, monkeypatch):
    """Start the ops routes on a free port, announce them, and watch the call go there."""
    import urllib.error
    from flask import Flask
    from werkzeug.serving import make_server

    monkeypatch.delenv("ARCANA_NO_SERVER")
    app = Flask("test-app")
    server.register(app)
    http = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        info = {"port": http.server_port, "pid": -1, "token": server._TOKEN}
        with open(server.discovery_path(), "w", encoding="utf-8") as fh:
            json.dump(info, fh)
        seen = []
        real = server.remote_exec
        monkeypatch.setattr(server, "remote_exec",
                            lambda *a, **k: seen.append(a[1]) or real(*a, **k))
        out = srv.run(['board create "Over the wire" --id wire', "board add wire synth:0"])
        assert seen == [['board create "Over the wire" --id wire', "board add wire synth:0"]]
        assert "Created `wire`" in texts(out)
        assert os.path.exists(os.path.join(tmp, "boards", "wire.json"))

        # A wrong token is shut out.
        with pytest.raises(urllib.error.HTTPError):
            real(dict(info, token="nope"), ["dataset list"], "agent")
    finally:
        http.shutdown()


def test_a_dead_server_record_is_ignored(tmp, ds, monkeypatch):
    monkeypatch.delenv("ARCANA_NO_SERVER")
    with open(server.discovery_path(), "w", encoding="utf-8") as fh:
        json.dump({"port": 1, "pid": 999999, "token": "x"}, fh)
    assert server.find(timeout=0.2) is None
    assert "Created" in texts(srv.run(['board create "Local" --id local']))


# ─────────────────────────── over a real pipe ───────────────────────────

# Starts the real server, except that every batch first prints to stdout the
# way db.py's progress messages do. On a pipe that line would be read as a
# malformed protocol message unless the transport keeps fd 1 off the wire.
_NOISY_SERVER = (
    "import arcana.server as s\n"
    "real = s.run_lines\n"
    "def noisy(*a, **k):\n"
    "    print('Encoding 3/10 ...', flush=True)\n"
    "    return real(*a, **k)\n"
    "s.run_lines = noisy\n"
    "import arcana.mcp_server as m\n"
    "m.main()\n"
)


def test_the_server_answers_over_stdio_even_when_an_op_prints(tmp, ds):
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ, ARCANA_DATA_DIR=tmp, PYTHONPATH=ROOT, ARCANA_NO_SERVER="1")
    params = StdioServerParameters(command=sys.executable, args=["-c", _NOISY_SERVER],
                                   env=env, cwd=ROOT)

    async def talk():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                ok = await session.call_tool("exec", {"commands": ["dataset list"]})
                wrong = await session.call_tool("exec", {"commands": ["search text nope --phrase x"]})
                return [t.name for t in tools.tools], ok, wrong

    names, ok, wrong = anyio.run(talk)
    assert names == ["exec"]
    assert not ok.is_error and "synth" in ok.content[0].text
    assert wrong.is_error and "Known datasets: synth" in wrong.content[0].text
