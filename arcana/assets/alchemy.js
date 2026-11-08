/* The Alchemy stage (ui_alchemy.py): the before/after line and the load bar.
 *
 * The line. Dragging anywhere on the picture moves it. The position lives in
 * the CSS variable --cut on the .alc-compare box, which Dash never sets, so a
 * new preview keeps the line where it was. Nothing here goes through a
 * callback, so the line follows the pointer at frame rate.
 *
 * The bar. Dash marks an element data-dash-is-loading while the newest request
 * for it is out, and clears the mark in the same update that brings the new
 * picture. The bar follows that mark on the stage image (a preview) and on the
 * Keep status (a Keep). It fills over the time the same step took last time,
 * keyed by method, view and size, and completes as the picture lands. A step
 * expected to take less than REALTIME ms shows no bar, unless it runs late.
 */
(function () {
  "use strict";

  // ── the before/after line ──
  var dragging = null;

  function place(box, clientX) {
    var r = box.getBoundingClientRect();
    if (!r.width) return;
    var x = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    box.style.setProperty("--cut", (x * 100).toFixed(2) + "%");
  }

  document.addEventListener("pointerdown", function (e) {
    var box = e.target.closest && e.target.closest(".alc-compare");
    if (!box || e.button !== 0) return;
    dragging = box;
    try { box.setPointerCapture(e.pointerId); } catch (err) { /* old browsers */ }
    place(box, e.clientX);
    e.preventDefault();
  });
  document.addEventListener("pointermove", function (e) {
    if (dragging) place(dragging, e.clientX);
  });
  function stop() { dragging = null; }
  document.addEventListener("pointerup", stop);
  document.addEventListener("pointercancel", stop);

  // ── the load bar ──
  var REALTIME = 450;        // ms: a step expected faster than this shows no bar
  var LATE = 150;            // ms: how long a step of unknown length waits before showing it
  var STORE = "arcana.alchemy.timings.v1";
  // The element whose loading mark drives the bar, and the element whose title names the step.
  var WATCH = { "alc-image": "alc-stage-meta", "alc-keep-status": "alc-keep-meta" };

  var took = {};
  try { took = JSON.parse(localStorage.getItem(STORE) || "{}") || {}; } catch (err) { took = {}; }
  var jobs = {};

  function bar() { return document.getElementById("alc-loadbar"); }
  function fill() { var b = bar(); return b ? b.querySelector(".alc-loadbar-fill") : null; }
  function stepKey(id) {
    var m = document.getElementById(WATCH[id]);
    return (m && m.title) || id;
  }
  function scaleNow(f) {
    var t = getComputedStyle(f).transform;
    var m = /matrix\(([^,]+)/.exec(t || "");
    return m ? parseFloat(m[1]) : 0;
  }
  function shownJob(except) {
    for (var id in jobs) if (id !== except && jobs[id].shown) return jobs[id];
    return null;
  }

  function begin(id) {
    if (jobs[id]) return;
    var key = stepKey(id);
    var job = jobs[id] = { start: performance.now(), key: key, expect: took[key], shown: false, done: false };
    var delay = job.expect === undefined ? LATE : (job.expect < REALTIME ? REALTIME + LATE : 0);
    job.timer = setTimeout(function () { show(job); }, delay);
  }

  function show(job) {
    var b = bar(), f = fill();
    if (!b || !f || job.done) return;
    job.shown = true;
    var expect = job.expect || 2500;
    var elapsed = performance.now() - job.start;
    var from = Math.max(scaleNow(f), Math.min(0.8, 0.92 * elapsed / expect));
    f.getAnimations().forEach(function (a) { a.cancel(); });
    b.classList.add("on");
    var first = f.animate([{ transform: "scaleX(" + from + ")" }, { transform: "scaleX(0.92)" }],
                          { duration: Math.max(200, expect - elapsed), easing: "cubic-bezier(0.3, 0.55, 0.45, 1)",
                            fill: "forwards" });
    // Past the expected time, the last stretch creeps rather than stopping dead. Timers, not
    // animation events, drive the steps, so the bar behaves the same in a background tab.
    job.creep = setTimeout(function () {
      if (job.done) return;
      first.cancel();
      f.animate([{ transform: "scaleX(0.92)" }, { transform: "scaleX(0.985)" }],
                { duration: expect * 4, easing: "ease-out", fill: "forwards" });
    }, Math.max(200, expect - elapsed));
  }

  function end(id) {
    var job = jobs[id];
    if (!job) return;
    delete jobs[id];
    job.done = true;
    clearTimeout(job.timer);
    clearTimeout(job.creep);
    var ms = performance.now() - job.start;
    // The latest time, not an average: a model's first load takes many times its later steps, and
    // the next step is the one the bar must time.
    took[job.key] = ms;
    try { localStorage.setItem(STORE, JSON.stringify(took)); } catch (err) { /* private window */ }
    if (!job.shown || shownJob(id)) return;
    var b = bar(), f = fill();
    if (!b || !f) return;
    var cur = scaleNow(f);
    f.getAnimations().forEach(function (a) { a.cancel(); });
    f.animate([{ transform: "scaleX(" + cur + ")" }, { transform: "scaleX(1)" }],
              { duration: 140, easing: "ease-out", fill: "forwards" });
    setTimeout(function () {
      if (shownJob()) return;
      b.classList.remove("on");
      setTimeout(function () {
        if (shownJob()) return;
        f.getAnimations().forEach(function (a) { a.cancel(); });
      }, 220);
    }, 160);
  }

  // ── the gate: one preview out at a time (ui_alchemy.py) ──
  // ask() is called by the gate's clientside callback on every change: it sends a request when none
  // is out, and otherwise holds the change. release() runs when the stage image's request lands and
  // sends the held change, so the newest settings are always the next ones drawn.
  var gate = window.alcGate = {
    out: false, held: false, draw: false, seq: 0, at: 0,
    ask: function (draw) {
      if (draw) this.draw = true;
      if (this.out && Date.now() - this.at < 180000) { this.held = true; return window.dash_clientside.no_update; }
      return this.send();
    },
    send: function () {
      this.held = false; this.out = true; this.at = Date.now(); this.seq += 1;
      var req = { seq: this.seq, draw: this.draw };
      this.draw = false;
      return req;
    },
    release: function () {
      this.out = false;
      if (this.held && window.dash_clientside && window.dash_clientside.set_props) {
        window.dash_clientside.set_props("alc-request", { data: this.send() });
      }
    }
  };

  new MutationObserver(function (records) {
    records.forEach(function (r) {
      var id = r.target.id;
      if (!(id in WATCH)) return;
      if (r.target.getAttribute("data-dash-is-loading") === "true") { begin(id); return; }
      end(id);
      if (id === "alc-image") gate.release();
    });
  }).observe(document.documentElement, { subtree: true, attributes: true, attributeFilter: ["data-dash-is-loading"] });
})();
