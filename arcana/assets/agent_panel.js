// agent_panel.js -- the Agent tab's terminal
//
// Dash renders the tab's frame (ui_agent.py). This fills it: a launcher for the
// agents found on this machine, a bar of the running ones, and an xterm.js
// terminal on the one shown. Output arrives over Server-Sent Events, keys go
// back as small POSTs (arcana/agent_term.py). Each agent's terminal lives here,
// outside Dash, so switching tabs or agents keeps its scrollback, and a page
// reload replays it from the server.
//
// The page checks in every 20 seconds from any tab, which keeps its agents
// running (agent_term.py stops them when no page has checked in for a few
// minutes). An agent that exits leaves the server's roster at once. Its last
// screen stays here until it is closed, and a reload clears it.
//
// xterm.js and its fit add-on are vendored in assets/vendor (MIT). Dash loads
// every script in assets/, in an order this does not rely on: it waits for
// window.Terminal before drawing anything.

(function () {
  "use strict";

  var S = {
    panel: null,        // #agent-panel, once Dash has rendered it
    host: null,         // #agent-term-host
    terms: {},          // instance id -> {term, fit, es, size}
    current: null,      // the instance on screen
    roster: { offered: [], instances: [], workspace: "" },
  };

  function token() { return S.panel ? S.panel.getAttribute("data-token") : ""; }

  function api(path, body) {
    var opts = { method: body === undefined ? "GET" : "POST",
                 headers: { "X-Arcana-Term": token(), "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    return fetch(path, opts).then(function (r) { return r.json(); });
  }

  function visible() {
    return S.panel && S.panel.offsetParent !== null && S.panel.style.display !== "none";
  }

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    for (var k in (attrs || {})) {
      if (k === "onclick") e.onclick = attrs[k]; else e.setAttribute(k, attrs[k]);
    }
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // ── the roster: what can start, and what is running ──
  function refresh() {
    return api("/agent/roster").then(function (r) {
      // Agents that exited while this page watched them: the server has let
      // them go, and their last screen stays here until closed.
      var listed = r.instances.map(function (i) { return i.id; });
      Object.keys(S.terms).forEach(function (id) {
        if (listed.indexOf(id) < 0) {
          r.instances.push({ id: id, name: S.terms[id].name, running: false });
        }
      });
      S.roster = r;
      var alive = r.instances.map(function (i) { return i.id; });
      if (S.current && alive.indexOf(S.current) < 0) S.current = null;
      if (!S.current) {
        var running = r.instances.filter(function (i) { return i.running; });
        if (running.length) show(running[0].id);
      }
      render();
    }).catch(function () {});
  }

  function render() {
    var bar = document.getElementById("agent-bar");
    var launch = document.getElementById("agent-launcher");
    var where = document.getElementById("agent-where");
    if (!bar || !launch) return;
    if (where) where.textContent = S.roster.workspace ? "Works in " + S.roster.workspace : "";

    bar.replaceChildren();
    S.roster.instances.forEach(function (i) {
      var tab = el("span", { "class": "agent-tab" + (i.id === S.current ? " on" : "") });
      tab.appendChild(el("button", { "class": "agent-tab-name", "data-testid": "agent-tab",
                                     onclick: function () { show(i.id); } },
                         i.name + (i.running ? "" : " (exited)")));
      tab.appendChild(el("button", { "class": "agent-tab-close", title: i.running ? "Stop this agent" : "Close",
                                     onclick: function () { close(i); } }, "×"));
      bar.appendChild(tab);
    });
    var add = el("button", { "class": "agent-tab-add", title: "Start another agent",
                             onclick: function () { S.current = null; render(); } }, "+");
    bar.appendChild(add);

    var showing = !!S.current;
    S.host.style.display = showing ? "block" : "none";
    launch.style.display = showing ? "none" : "flex";
    if (!showing) {
      launch.replaceChildren();
      var box = el("div", { "class": "agent-launch-box" });
      box.appendChild(el("div", { "class": "agent-launch-title" }, "Start an agent"));
      box.appendChild(el("div", { "class": "agent-launch-hint" },
        S.roster.offered.length
          ? "It runs in the Arcana workspace with Arcana's tools and skills, and acts as an agent: " +
            "it changes only boards it made, and sees thumbnails only if you allow it."
          : "No agent command line was found on this machine. Install Claude Code, or list one " +
            "in agents.json in Arcana's data folder."));
      var grid = el("div", { "class": "agent-launch-grid" });
      S.roster.offered.forEach(function (a) {
        grid.appendChild(el("button", { "class": "agent-launch", "data-testid": "agent-launch",
                                        title: a.command, onclick: function () { start(a.name); } },
                            a.name));
      });
      box.appendChild(grid);
      launch.appendChild(box);
    } else {
      fit();
    }
  }

  function start(name) {
    var d = measure() || { cols: 120, rows: 32 };
    api("/agent/start", { name: name, cols: d.cols, rows: d.rows }).then(function (info) {
      if (info.error) { alert(info.error); return; }
      S.current = info.id;
      return refresh().then(function () { show(info.id); });
    });
  }

  function discard(id) {
    var t = S.terms[id];
    if (t) { if (t.es) t.es.close(); t.term.dispose(); delete S.terms[id]; }
    if (S.current === id) S.current = null;
  }

  function close(i) {
    if (i.running) {
      if (!confirm("Stop " + i.name + "? Its session ends, and the workspace stays.")) return;
      api("/agent/" + i.id + "/stop", {}).then(function () { discard(i.id); refresh(); });
    } else {
      api("/agent/" + i.id + "/forget", {}).then(function () { discard(i.id); refresh(); });
    }
  }

  // ── terminals ──
  function terminalFor(id) {
    var t = S.terms[id];
    if (t) return t;
    var term = new window.Terminal({
      fontFamily: '"Cascadia Mono", "Cascadia Code", Consolas, "Courier New", monospace',
      fontSize: 13, scrollback: 10000, cursorBlink: true, allowProposedApi: true,
      theme: { background: "#141311", foreground: "#ede8df", cursor: "#c8743f",
               selectionBackground: "#c8743f55", black: "#1c1a17", red: "#b5423a",
               green: "#7a9a5e", yellow: "#d6a74d", blue: "#7c8f8a", magenta: "#a37a8a",
               cyan: "#8c9a5b", white: "#ede8df" },
    });
    var fitter = new window.FitAddon.FitAddon();
    term.loadAddon(fitter);
    var pending = "", timer = null, replaying = false;
    term.onData(function (data) {
      // A replayed history holds the agent's old terminal queries. xterm
      // answers them, and those answers must not reach the agent as keys.
      if (replaying) return;
      // Batch a burst of keys (a paste) into one request.
      pending += data;
      if (!timer) timer = setTimeout(function () {
        var out = pending; pending = ""; timer = null;
        api("/agent/" + id + "/input", { data: out });
      }, 8);
    });
    var known = S.roster.instances.filter(function (i) { return i.id === id; })[0];
    t = { term: term, fit: fitter, es: null, size: "", name: known ? known.name : "Agent",
          el: el("div", { "class": "agent-term" }) };
    term.open(t.el);
    var es = new EventSource("/agent/" + id + "/stream?token=" + encodeURIComponent(token()));
    es.onmessage = function (e) {
      var m = JSON.parse(e.data);
      if (m.d && m.replay) {
        replaying = true;
        term.write(m.d, function () { replaying = false; });
      } else if (m.d) {
        term.write(m.d);
      }
      if (m.exit !== undefined) {
        term.write("\r\n\x1b[2m[the agent exited with code " + m.exit + "]\x1b[0m\r\n");
        es.close();
        refresh();
      }
    };
    t.es = es;
    S.terms[id] = t;
    return t;
  }

  function show(id) {
    if (!window.Terminal || !window.FitAddon) { setTimeout(function () { show(id); }, 100); return; }
    S.current = id;
    var t = terminalFor(id);
    S.host.replaceChildren(t.el);
    render();
    setTimeout(function () { fit(); t.term.focus(); }, 30);
  }

  function measure() {
    var t = S.current && S.terms[S.current];
    if (!t) return null;
    var d = t.fit.proposeDimensions();
    return d && d.cols && d.rows ? d : null;
  }

  function fit() {
    var t = S.current && S.terms[S.current];
    if (!t || !visible()) return;
    var d = measure();
    if (!d) return;
    t.fit.fit();
    var key = d.cols + "x" + d.rows;
    if (key !== t.size) {
      t.size = key;
      api("/agent/" + S.current + "/resize", { cols: d.cols, rows: d.rows });
    }
  }

  // ── wiring: wait for Dash to render the tab, then follow its visibility ──
  function boot() {
    var panel = document.getElementById("agent-panel");
    var host = document.getElementById("agent-term-host");
    if (!panel || !host) { setTimeout(boot, 300); return; }
    S.panel = panel;
    S.host = host;
    var wasVisible = false;
    new MutationObserver(function () {
      var now = visible();
      if (now && !wasVisible) refresh();
      wasVisible = now;
    }).observe(panel, { attributes: true, attributeFilter: ["style"] });
    new ResizeObserver(function () { fit(); }).observe(host);
    if (visible()) { wasVisible = true; refresh(); }
    // Check in from whichever tab is showing, so the agents know a page is open.
    function checkIn() {
      if (visible()) refresh();
      else api("/agent/roster").catch(function () {});
    }
    checkIn();
    setInterval(checkIn, 20000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
