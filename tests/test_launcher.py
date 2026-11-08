"""
The installed build's launcher: modes are flags on one executable.

`Arcana.exe --mcp` serves an agent over stdio, so stdout must stay the
protocol's and only stderr may go to the log. `--headless` serves the ops
without the pages. Plain `Arcana.exe` announces itself in server.json so the
terminal and an agent reach it. These run the launcher's main() with the
servers replaced by stubs.

Run with:  python -m pytest tests/test_launcher.py -q
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def launcher(tmp, monkeypatch):
    monkeypatch.setenv("ARCANA_DATA_DIR", tmp)
    spec = importlib.util.spec_from_file_location("arcana_launcher",
                                                  os.path.join(ROOT, "installer", "launcher.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_frozen", lambda: True)
    return mod


def test_mcp_mode_keeps_stdout_for_the_protocol(launcher, monkeypatch, tmp):
    from arcana import mcp_server
    seen = {}
    monkeypatch.setattr(mcp_server, "main", lambda: seen.setdefault("stdout", sys.stdout))
    real_stdout, real_stderr = sys.stdout, sys.stderr
    try:
        assert launcher.main(["--mcp"]) == 0
        assert seen["stdout"] is real_stdout
        assert sys.stderr is not real_stderr and sys.stderr.name.endswith("arcana-mcp.log")
    finally:
        log = sys.stderr
        sys.stdout, sys.stderr = real_stdout, real_stderr
        if log is not real_stderr:
            log.close()


def test_headless_mode_serves_the_ops(launcher, monkeypatch):
    from arcana import server
    ports = []
    monkeypatch.setattr(server, "serve_headless", lambda port: ports.append(port))
    real_stdout, real_stderr = sys.stdout, sys.stderr
    try:
        assert launcher.main(["--headless", "--port", "8123"]) == 0
    finally:
        for s in (sys.stdout,):
            if s is not real_stdout:
                s.close()
        sys.stdout, sys.stderr = real_stdout, real_stderr
    assert ports == [8123]


def test_the_spec_ships_the_ops_the_sdk_and_the_skills():
    text = open(os.path.join(ROOT, "installer", "arcana.spec"), encoding="utf-8").read()
    for name in ("arcana.ops", "arcana.mcp_server", "arcana.server", "arcana.state",
                 "arcana.compose", "arcana.generate", "arcana.transfer_methods"):
        assert f'"{name}"' in text, name
    assert 'collect_submodules("mcp"' in text
    assert 'not name.startswith("mcp.cli")' in text      # the SDK's own CLI needs typer
    assert '"plugin", "skills"' in text


def test_mcp_is_a_main_dependency():
    reqs = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read()
    assert "\nmcp>=" in reqs


def test_version_answers_without_starting_the_app(launcher, capsys):
    assert launcher.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("Arcana ")
