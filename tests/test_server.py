"""
The ops over HTTP: the routes the CLI and the MCP server forward to while the
app is open, through Flask's test client.

Run with:  python -m pytest tests/test_server.py -q
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from arcana import server, state  # noqa: E402

import synth  # noqa: E402


@pytest.fixture
def client(tmp, monkeypatch):
    from flask import Flask
    synth.use(tmp, monkeypatch)
    app = Flask("t")
    server.register(app)
    server.register(app)                 # idempotent
    c = app.test_client()
    c.environ_base["HTTP_X_ARCANA_TOKEN"] = server._TOKEN
    return c


def test_every_route_needs_the_token(client):
    bare = client.application.test_client()
    for method, route in (("get", "/ops/list"), ("get", "/ops/state"), ("post", "/ops/exec")):
        assert getattr(bare, method)(route).status_code == 403


def test_exec_runs_a_batch_and_bumps_the_version_once(client):
    v0 = client.get("/ops/state").get_json()["version"]
    res = client.post("/ops/exec", json={"lines": ['board create "Rain" --id rain',
                                                   "board add rain synth:0", "board show rain"],
                                         "actor": "agent"}).get_json()
    assert res["ok"] and "Created `rain`" in res["text"]
    assert res["steps"][2]["result"]["metrics"]["count"] == 1
    assert client.get("/ops/state").get_json()["version"] == v0 + 1


def test_a_refusal_is_an_answer_not_an_http_error(client):
    r = client.post("/ops/exec", json={"lines": ["search text nope --phrase x"], "actor": "cli"})
    assert r.status_code == 200
    body = r.get_json()
    assert not body["ok"] and "Known datasets" in body["steps"][0]["refusal"]["reason"]


def test_nobody_claims_to_be_the_person_over_http(client):
    r = client.post("/ops/exec", json={"lines": ["board list"], "actor": "person"})
    assert r.status_code == 400


def test_rpc_takes_one_op_as_json(client):
    body = client.post("/ops/rpc", json={"op": "search text", "actor": "agent",
                                         "args": {"dataset": "synth", "phrase": ["water"], "k": 2}}).get_json()
    assert body["op"] == "search text" and len(body["result"]["facets"][0]["hits"]) == 2
    bad = client.post("/ops/rpc", json={"op": "search text", "actor": "agent", "args": {"k": 2}}).get_json()
    assert "--phrase is required" in bad["refused"]["reason"]


def test_pictures_travel_only_when_the_person_allows_them(client, tmp):
    lines = {"lines": ["item sheet synth:1 synth:2"], "actor": "agent"}
    blind = client.post("/ops/exec", json=lines).get_json()
    assert blind["steps"][0]["attachments"] == []
    synth.pixels(tmp, True)
    seen = client.post("/ops/exec", json=lines).get_json()
    assert len(seen["steps"][0]["attachments"]) == 1


def test_the_op_list_and_schema(client):
    names = {o["name"] for o in client.get("/ops/list").get_json()}
    assert "board add" in names and "search text" in names
    schema = client.get("/ops/schema.json").get_json()
    assert schema["board add"]["kind"] == "write"


def test_announce_writes_and_withdraws_discovery(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_DATA_DIR", tmp)
    path = server.announce(5555)
    info = json.load(open(path))
    assert info["port"] == 5555 and info["pid"] == os.getpid() and info["token"] == server._TOKEN
    # This process never forwards to itself.
    assert server.find() is None
