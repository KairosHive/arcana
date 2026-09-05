/* Draggable column dividers for the moodboard.
 *
 * The bench is three columns -- collection, tool, output -- at widths chosen
 * once and never adjustable, so seeing a collection thumbnail larger meant
 * asking for a bigger thumbnail. Dragging the boundary is the general answer.
 *
 * Two things make this less obvious than it sounds:
 *
 *   1. Dash writes the columns' `flex` as an INLINE style, and rewrites it on
 *      every mode change. Anything this script set directly would be wiped the
 *      next time the user switched tools. So the widths live in CSS custom
 *      properties, and a rule in custom.css binds them with !important, which
 *      an inline style cannot beat. This script only ever moves the variables.
 *
 *   2. The handles are NOT inserted into #main-row. That container belongs to
 *      React, which is entitled to replace its children at any point and would
 *      take the handles with them. They are positioned over the boundaries from
 *      document.body instead, so the two trees never touch.
 */
(function () {
  "use strict";

  var RAIL = "--arc-rail-w";
  var BENCH = "--arc-bench-w";
  var STORE = "arcana.columns.v1";
  var MIN_RAIL = 140, MAX_RAIL = 560;
  var MIN_BENCH = 220, MAX_BENCH_FRACTION = 0.6;

  function readSaved() {
    try {
      var raw = window.localStorage.getItem(STORE);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;               // private window, or site data blocked
    }
  }

  function save(rail, bench) {
    try {
      window.localStorage.setItem(STORE, JSON.stringify({ rail: rail, bench: bench }));
    } catch (e) { /* a lost layout preference is not worth an error */ }
  }

  var saved = readSaved();
  if (saved) {
    if (saved.rail) document.documentElement.style.setProperty(RAIL, saved.rail + "px");
    if (saved.bench) document.documentElement.style.setProperty(BENCH, saved.bench + "px");
  }

  function el(id) { return document.getElementById(id); }

  function makeHandle(key) {
    var h = document.createElement("div");
    h.className = "arc-splitter";
    h.dataset.key = key;
    h.setAttribute("role", "separator");
    h.setAttribute("aria-orientation", "vertical");
    h.title = "Drag to resize · double-click to reset";
    document.body.appendChild(h);
    return h;
  }

  var handles = { rail: makeHandle("rail"), bench: makeHandle("bench") };
  var dragging = null;

  function place() {
    var row = el("main-row");
    var rail = el("moodboard-rail");
    var left = el("left-column");
    // Only meaningful when all three columns are actually on screen.
    var live = row && rail && left &&
               getComputedStyle(row).display !== "none" &&
               getComputedStyle(rail).display !== "none" &&
               rail.getBoundingClientRect().width > 0;
    if (!live) {
      handles.rail.style.display = "none";
      handles.bench.style.display = "none";
      return;
    }
    var rb = rail.getBoundingClientRect();
    var lb = left.getBoundingClientRect();
    [["rail", rb.right], ["bench", lb.right]].forEach(function (pair) {
      var h = handles[pair[0]];
      h.style.display = "block";
      h.style.top = rb.top + "px";
      h.style.height = rb.height + "px";
      h.style.left = (pair[1] - 3) + "px";
    });
  }

  function onDown(e) {
    var h = e.target.closest(".arc-splitter");
    if (!h) return;
    var rail = el("moodboard-rail"), left = el("left-column");
    if (!rail || !left) return;
    dragging = {
      key: h.dataset.key,
      startX: e.clientX,
      rail: rail.getBoundingClientRect().width,
      bench: left.getBoundingClientRect().width,
    };
    document.body.classList.add("arc-resizing");
    e.preventDefault();
  }

  function onMove(e) {
    if (!dragging) return;
    var dx = e.clientX - dragging.startX;
    var row = el("main-row");
    var maxBench = row ? row.getBoundingClientRect().width * MAX_BENCH_FRACTION : 900;

    if (dragging.key === "rail") {
      var w = Math.max(MIN_RAIL, Math.min(MAX_RAIL, dragging.rail + dx));
      document.documentElement.style.setProperty(RAIL, w + "px");
    } else {
      var b = Math.max(MIN_BENCH, Math.min(maxBench, dragging.bench + dx));
      document.documentElement.style.setProperty(BENCH, b + "px");
    }
    place();
  }

  function currentPx(name) {
    var v = document.documentElement.style.getPropertyValue(name);
    return v ? parseInt(v, 10) : null;
  }

  function onUp() {
    if (!dragging) return;
    dragging = null;
    document.body.classList.remove("arc-resizing");
    save(currentPx(RAIL), currentPx(BENCH));
  }

  function onDouble(e) {
    var h = e.target.closest(".arc-splitter");
    if (!h) return;
    // Back to the layout's own defaults, by removing the override entirely.
    document.documentElement.style.removeProperty(h.dataset.key === "rail" ? RAIL : BENCH);
    save(currentPx(RAIL), currentPx(BENCH));
    place();
  }

  document.addEventListener("mousedown", onDown);
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
  document.addEventListener("dblclick", onDouble);
  window.addEventListener("resize", place);

  // Dash re-renders the columns whenever the mode or tool changes, which moves
  // the boundaries without any event this script would otherwise hear.
  if (window.ResizeObserver) {
    var ro = new ResizeObserver(place);
    var attach = setInterval(function () {
      var row = el("main-row");
      if (row) { ro.observe(row); clearInterval(attach); }
    }, 400);
  }
  setInterval(place, 700);
  place();
})();
