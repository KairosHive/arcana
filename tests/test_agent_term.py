"""
The Agent tab's backend: the workspace, the terminal manager and its routes.

Agents here are short Python programs listed in agents.json, run in a real
pseudo-terminal, so the tests need no agent installed. They check that an
agent works in the workspace as the agent actor, that keys reach it and its
output streams back, and that nothing reaches the terminal without the page's
token.

Run with:  python -m pytest tests/test_agent_term.py -q
"""

import json
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

pty = pytest.importorskip("winpty" if sys.platform == "win32" else "ptyprocess")

from arcana import agent_term, paths, workspace  # noqa: E402


@pytest.fixture
def data(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_DATA_DIR", tmp)
    return tmp


def _agents(tmp, *entries):
    with open(os.path.join(tmp, "agents.json"), "w", encoding="utf-8") as fh:
        json.dump({"agents": list(entries)}, fh)


def _py(name, code):
    return {"name": name, "command": sys.executable, "args": ["-u", "-c", code]}


def _wait_for(inst, text, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if text in "".join(inst.text):
            return True
        time.sleep(0.1)
    return False


# ─────────────────────────── the workspace ───────────────────────────

def test_the_workspace_sits_beside_the_output(data):
    out = paths.subdir("output")
    assert os.path.dirname(workspace.workspace_dir()) == os.path.dirname(os.path.abspath(out))


def test_prepare_writes_the_guide_the_server_and_the_skills(data):
    ws = workspace.prepare()
    guide = open(os.path.join(ws, "CLAUDE.md"), encoding="utf-8").read()
    assert 'exec(["orient"])' in guide and "board copy" in guide
    assert open(os.path.join(ws, "AGENTS.md"), encoding="utf-8").read() == guide
    server = json.load(open(os.path.join(ws, ".mcp.json")))["mcpServers"]["arcana"]
    assert server["command"] == sys.executable and server["args"][-1] == "arcana.mcp_server"
    assert server["env"]["ARCANA_DATA_DIR"] == data
    settings = json.load(open(os.path.join(ws, ".claude", "settings.json")))
    assert "mcp__arcana__exec" in settings["permissions"]["allow"]
    assert "arcana" in settings["enabledMcpjsonServers"]
    for name in workspace.SKILLS:
        assert os.path.isfile(os.path.join(ws, ".claude", "skills", name, "SKILL.md"))


def test_prepare_keeps_what_the_person_wrote(data):
    ws = workspace.prepare()
    with open(os.path.join(ws, "CLAUDE.local.md"), "w", encoding="utf-8") as fh:
        fh.write("my notes")
    s = os.path.join(ws, ".claude", "settings.json")
    cfg = json.load(open(s))
    cfg["model"] = "opus"
    cfg["permissions"]["allow"].append("Bash(ls:*)")
    json.dump(cfg, open(s, "w"))
    workspace.prepare()
    assert open(os.path.join(ws, "CLAUDE.local.md"), encoding="utf-8").read() == "my notes"
    again = json.load(open(s))
    assert again["model"] == "opus" and "Bash(ls:*)" in again["permissions"]["allow"]
    assert again["permissions"]["allow"].count("mcp__arcana__exec") == 1


# ─────────────────────────── agents in a terminal ───────────────────────────

def test_an_agent_runs_in_the_workspace_as_the_agent(data):
    _agents(data, _py("Probe", "import os; print('cwd=' + os.getcwd()); "
                               "print('actor=' + os.environ['ARCANA_ACTOR'])"))
    assert "Probe" in [a["name"] for a in agent_term.offered()]
    inst = agent_term.MANAGER.start("Probe")
    try:
        assert _wait_for(inst, "actor=agent")
        out = "".join(inst.text).replace("\r", "").replace("\n", "")
        assert "cwd=" + workspace.workspace_dir() in out
        deadline = time.time() + 10
        while inst.running and time.time() < deadline:
            time.sleep(0.1)
        assert inst.exit_code == 0
    finally:
        agent_term.MANAGER.forget(inst.id)


def test_keys_reach_the_agent_and_its_answer_streams_back(data):
    _agents(data, _py("Echo", "s = input('> '); print('GOT ' + s.upper())"))
    inst = agent_term.MANAGER.start("Echo")
    try:
        history, q = inst.subscribe()
        assert _wait_for(inst, "> ")
        inst.write("abc\r\n")
        assert _wait_for(inst, "GOT ABC")
        seen = []
        while not q.empty():
            seen.append(q.get())
        assert any(k == "d" for k, _ in seen)
    finally:
        agent_term.MANAGER.forget(inst.id)


def test_an_unknown_agent_is_refused(data):
    with pytest.raises(ValueError, match="No agent called"):
        agent_term.MANAGER.start("Nope")


# ─────────────────────────── the routes ───────────────────────────

@pytest.fixture
def client(data):
    from flask import Flask
    app = Flask("t")
    agent_term.register(app)
    return app.test_client()


def test_nothing_reaches_the_terminal_without_the_pages_token(client, data):
    _agents(data, _py("Probe", "print('x')"))
    assert client.get("/agent/roster").status_code == 403
    assert client.post("/agent/start", json={"name": "Probe"}).status_code == 403
    assert client.post("/agent/start", json={"name": "Probe"},
                       headers={agent_term.HEADER: "wrong"}).status_code == 403
    # A token in the query only opens the stream, never a command.
    assert client.post(f"/agent/start?token={agent_term.TOKEN}",
                       json={"name": "Probe"}).status_code == 403


def test_start_stream_and_stop_through_the_routes(client, data):
    _agents(data, _py("Talk", "print('hello from the agent'); input()"))
    h = {agent_term.HEADER: agent_term.TOKEN}
    roster = client.get("/agent/roster", headers=h).get_json()
    assert "Talk" in [a["name"] for a in roster["offered"]]
    assert roster["workspace"] == workspace.workspace_dir()
    info = client.post("/agent/start", json={"name": "Talk", "cols": 100, "rows": 30}, headers=h).get_json()
    inst = agent_term.MANAGER.get(info["id"])
    try:
        assert _wait_for(inst, "hello from the agent")
        assert client.post(f"/agent/{info['id']}/resize", json={"cols": 90, "rows": 25},
                           headers=h).get_json()["ok"]
        assert inst.size == (90, 25)
        resp = client.get(f"/agent/{info['id']}/stream?token={agent_term.TOKEN}", buffered=False)
        first = next(resp.response)
        first = first.decode() if isinstance(first, bytes) else first
        assert first.startswith("data: ") and "hello from the agent" in first
        resp.close()
        client.post(f"/agent/{info['id']}/stop", json={}, headers=h)
        deadline = time.time() + 10
        while inst.running and time.time() < deadline:
            time.sleep(0.1)
        assert not inst.running
        listed = client.get("/agent/roster", headers=h).get_json()["instances"]
        assert [i["running"] for i in listed if i["id"] == info["id"]] == [False]
    finally:
        agent_term.MANAGER.forget(info["id"])


# ─────────────────────────── the terminal's own `arcana` ───────────────────────────

def test_the_command_line_in_the_agents_terminal_acts_as_the_agent(data, monkeypatch, capsys):
    sys.path.insert(0, HERE)
    import synth
    synth.use(data, monkeypatch)
    monkeypatch.setenv("ARCANA_NO_SERVER", "1")
    from arcana import cli
    assert cli.main(["board", "create", "Mine", "--id", "mine"]) == 0
    monkeypatch.setenv("ARCANA_ACTOR", "agent")
    assert cli.main(["board", "delete", "mine"]) == 1
    assert "for the person" in capsys.readouterr().err
    monkeypatch.setenv("ARCANA_ACTOR", "person")             # never claimable from a shell
    assert cli.main(["board", "add", "mine", "synth:0"]) == 0
