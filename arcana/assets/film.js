/* The Video subtab's timeline and player (ui_film.py, arcana/alchemy/film.py).
 *
 * The film lives in the Dash store film-spec. This script draws it as a timeline: a ruler with the
 * playhead, the picture lane (the film passes each picture at its key), the look lane (snapshots
 * from the Image subtab) and one lane per keyed number, drawn as its curve. Everything is edited by
 * hand here, and each finished edit writes film-spec back, so the panel, the player and a kept film
 * all read the same film.
 *
 *   drop a picture from the collection   put it on the picture lane where it lands
 *   drag the ruler or an empty lane      scrub
 *   drag a picture or a look             move it in time
 *   drag a lane's key                    move it in time and value
 *   double-click a lane                  add a key on its curve
 *   press a lane's \u25c6                  add a key at the playhead, or remove the one there
 *   click a key                          select it (the panel edits it); Delete removes it
 *   click a lane's curve                 select the key before that point, to set the curve's shape
 *   space                                play a draft, as fast as frames are drawn
 *
 * Frames are asked for one at a time (filmGate): a change while one is being drawn is held, and the
 * newest goes out when that frame lands.
 */
(function () {
  "use strict";

  var LABEL_W = 170, RULER = 22, PICS = 58, LOOKS = 26, LANE = 46, PAD = 8, HINT = 20;
  var EASES = { linear: function (s) { return s; }, smooth: function (s) { return s * s * (3 - 2 * s); },
                "in": function (s) { return s * s; }, out: function (s) { return 1 - (1 - s) * (1 - s); },
                hold: function () { return 0; } };
  var st = { spec: null, meta: null, t: 0, sel: null, drag: null, playing: false, dropT: null };

  function setProps(id, props) {
    if (window.dash_clientside && window.dash_clientside.set_props) window.dash_clientside.set_props(id, props);
  }
  function clone(x) { return JSON.parse(JSON.stringify(x)); }
  function sorted(keys) { return (keys || []).slice().sort(function (a, b) { return a.t - b.t; }); }
  function ease(name, s) { s = Math.min(Math.max(s, 0), 1); return (EASES[name] || EASES.linear)(s); }
  function laneValue(keys, t) {
    if (!keys.length) return 0;
    if (t <= keys[0].t) return keys[0].v;
    for (var i = 0; i + 1 < keys.length; i++) {
      var a = keys[i], b = keys[i + 1];
      if (t < b.t) return a.v + (b.v - a.v) * ease(a.ease, (t - a.t) / Math.max(b.t - a.t, 1e-9));
    }
    return keys[keys.length - 1].v;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function geom(el) {
    var W = Math.max(el.clientWidth, 320);
    var L = Math.max((st.meta && st.meta.length) || 1, 0.5);
    var x0 = LABEL_W + PAD, x1 = W - PAD;
    return { W: W, L: L, x0: x0, x1: x1,
             x: function (t) { return x0 + (t / L) * (x1 - x0); },
             t: function (x) { return Math.min(Math.max((x - x0) / (x1 - x0) * L, 0), L); } };
  }
  function lanes() { return Object.keys((st.spec && st.spec.lanes) || {}); }
  function laneTop(i) { return RULER + PICS + LOOKS + i * LANE; }

  function draw() {
    var el = document.getElementById("film-timeline");
    if (!el || !st.spec) return;
    var g = geom(el), spec = st.spec, meta = st.meta || {}, ls = lanes();
    var H = laneTop(ls.length) + HINT;
    var o = ['<svg width="' + g.W + '" height="' + H + '" class="film-svg">'];
    // ruler
    var step = [0.5, 1, 2, 5, 10, 15, 30, 60].find(function (s) { return (g.x1 - g.x0) / (g.L / s) >= 46; }) || 120;
    o.push('<rect class="film-ruler" data-kind="ruler" x="0" y="0" width="' + g.W + '" height="' + RULER + '"/>');
    for (var s = 0; s <= g.L + 1e-6; s += step) {
      var xx = g.x(s);
      o.push('<line class="film-tick" x1="' + xx + '" y1="' + (RULER - 7) + '" x2="' + xx + '" y2="' + RULER + '"/>');
      o.push('<text class="film-tick-text" x="' + (xx + 3) + '" y="12">' + (s >= 60 ? Math.floor(s / 60) + ":" + String(Math.round(s % 60)).padStart(2, "0") : s + "s") + "</text>");
    }
    // row labels and bands
    var rows = [["pictures", RULER, PICS, "pictures"], ["looks", RULER + PICS, LOOKS, "looks"]];
    ls.forEach(function (p, i) { rows.push([p, laneTop(i), LANE, ((meta.lanes || {})[p] || {}).label || p]); });
    rows.forEach(function (r, i) {
      o.push('<rect class="film-band' + (i % 2 ? " odd" : "") + '" data-kind="band" data-lane="' + esc(r[0]) + '" x="0" y="' + r[1] + '" width="' + g.W + '" height="' + r[2] + '"/>');
      o.push('<text class="film-row-label" x="10" y="' + (r[1] + Math.min(r[2] / 2 + 4, 16)) + '">' + esc(r[3]) + "</text>");
      if (r[0] !== "pictures" && r[0] !== "looks") {
        var keys0 = sorted(spec.lanes[r[0]]), here = keys0.some(function (k) { return Math.abs(k.t - st.t) < 0.02; });
        var stp = (((meta.lanes || {})[r[0]]) || {}).step || 0.01, dec = stp >= 1 ? 0 : stp >= 0.1 ? 1 : 2;
        o.push('<text class="film-row-value" x="10" y="' + (r[1] + 32) + '">' + laneValue(keys0, st.t).toFixed(dec) + "</text>");
        o.push('<text class="film-keyhere' + (here ? " on" : "") + '" data-kind="keyhere" data-lane="' + esc(r[0]) + '" x="' + (LABEL_W - 34) + '" y="' + (r[1] + 18) + '"><title>' + (here ? "Remove the key at the playhead" : "Add a key at the playhead") + "</title>\u25c6</text>");
        o.push('<text class="film-row-remove" data-kind="remove" data-lane="' + esc(r[0]) + '" x="' + (LABEL_W - 16) + '" y="' + (r[1] + 18) + '"><title>Remove this lane</title>\u00d7</text>');
      }
    });
    // pictures
    var pics = sorted(spec.pictures);
    pics.forEach(function (p, i) {
      var xx = g.x(p.t), sel = st.sel && st.sel.lane === "pictures" && st.sel.i === i;
      var th = ((meta.pictures || [])[i] || {}).thumb || "";
      o.push('<line class="film-key-line" x1="' + xx + '" y1="' + RULER + '" x2="' + xx + '" y2="' + (RULER + PICS) + '"/>');
      o.push('<image href="' + esc(th) + '" x="' + (xx - 22) + '" y="' + (RULER + 5) + '" width="44" height="44" preserveAspectRatio="xMidYMid slice" class="film-thumb" data-kind="key" data-lane="pictures" data-i="' + i + '"/>');
      o.push('<rect class="film-thumb-frame' + (sel ? " sel" : "") + '" x="' + (xx - 22) + '" y="' + (RULER + 5) + '" width="44" height="44" rx="4" pointer-events="none"/>');
    });
    // looks
    sorted(spec.looks).forEach(function (k, i) {
      var xx = g.x(k.t), y = RULER + PICS + LOOKS / 2, sel = st.sel && st.sel.lane === "looks" && st.sel.i === i;
      o.push('<path class="film-look' + (sel ? " sel" : "") + '" data-kind="key" data-lane="looks" data-i="' + i + '" d="M' + xx + "," + (y - 7) + " l7,7 l-7,7 l-7,-7z\"/>");
      var lab = ((meta.looks || [])[i] || {}).label || "";
      o.push('<text class="film-look-text" x="' + (xx + 10) + '" y="' + (y + 4) + '">' + esc(lab) + "</text>");
    });
    // lanes
    ls.forEach(function (p, li) {
      var keys = sorted(spec.lanes[p]), m = (meta.lanes || {})[p] || { lo: 0, hi: 1 }, top = laneTop(li);
      var lo = m.lo, hi = m.hi === m.lo ? m.lo + 1 : m.hi;
      var y = function (v) { return top + 6 + (1 - (v - lo) / (hi - lo)) * (LANE - 12); };
      var pts = [];
      for (var k = 0; k <= 160; k++) { var tt = g.L * k / 160; pts.push(g.x(tt).toFixed(1) + "," + y(laneValue(keys, tt)).toFixed(1)); }
      o.push('<polyline class="film-curve-hit" data-kind="curve" data-lane="' + esc(p) + '" points="' + pts.join(" ") + '"/>');
      o.push('<polyline class="film-curve" points="' + pts.join(" ") + '"/>');
      keys.forEach(function (key, i) {
        var sel = st.sel && st.sel.lane === p && st.sel.i === i;
        o.push('<circle class="film-dot' + (sel ? " sel" : "") + '" data-kind="key" data-lane="' + esc(p) + '" data-i="' + i + '" cx="' + g.x(key.t) + '" cy="' + y(key.v) + '" r="' + (sel ? 6 : 5) + '"/>');
      });
      o.push('<text class="film-lane-range" x="' + (LABEL_W - 4) + '" y="' + (top + LANE - 6) + '" text-anchor="end">' + (+lo.toFixed(2)) + "–" + (+hi.toFixed(2)) + "</text>");
    });
    // where a dragged picture would land
    if (st.dropT !== null) {
      var dx = g.x(st.dropT);
      o.push('<line class="film-drop" x1="' + dx + '" y1="' + RULER + '" x2="' + dx + '" y2="' + (RULER + PICS) + '"/>');
    }
    o.push('<text class="film-hint" x="10" y="' + (H - 6) + '">Drop pictures on the timeline \u00b7 double-click a lane or press \u25c6 to add a key \u00b7 drag keys \u00b7 click a curve to set its shape \u00b7 Delete removes</text>');
    // playhead
    var px = g.x(st.t);
    o.push('<line class="film-playhead" x1="' + px + '" y1="0" x2="' + px + '" y2="' + H + '"/>');
    o.push('<path class="film-playhead-cap" data-kind="ruler" d="M' + (px - 6) + ",0 h12 l-6,8z\"/>");
    o.push("</svg>");
    el.innerHTML = o.join("");
  }

  function commit(spec, sel) {
    st.spec = spec;
    setProps("film-spec", { data: spec });
    if (sel !== undefined) setProps("film-selected", { data: sel });
    draw();
  }

  function keyList(spec, lane) {
    if (lane === "pictures" || lane === "looks") { spec[lane] = sorted(spec[lane]); return spec[lane]; }
    spec.lanes[lane] = sorted(spec.lanes[lane]); return spec.lanes[lane];
  }

  function onDown(e) {
    var el = document.getElementById("film-timeline");
    var target = e.target, kind = target.getAttribute && target.getAttribute("data-kind");
    if (!kind || !st.spec || e.button !== 0) return;
    el.focus();
    var r = el.getBoundingClientRect(), g = geom(el), x = e.clientX - r.left;
    if (kind === "ruler") {
      st.drag = { kind: "ruler" };
      st.t = g.t(x); setProps("film-time", { data: st.t }); draw();
    } else if (kind === "keyhere") {
      var lk = target.getAttribute("data-lane"), sp = clone(st.spec), ks = keyList(sp, lk);
      var at = ks.findIndex(function (k) { return Math.abs(k.t - st.t) < 0.02; });
      if (at >= 0) { ks.splice(at, 1); commit(sp, null); return; }
      var nk = { t: Math.round(st.t * 100) / 100, v: laneValue(ks, st.t), ease: "linear" };
      ks.push(nk); ks = keyList(sp, lk);
      commit(sp, { lane: lk, i: ks.indexOf(nk) });
      return;
    } else if (kind === "curve") {
      var lc = target.getAttribute("data-lane"), kc = sorted(st.spec.lanes[lc]), tc = g.t(x), pick = -1;
      for (var q = 0; q < kc.length; q++) if (kc[q].t <= tc) pick = q;
      if (pick >= 0) { st.sel = { lane: lc, i: pick }; setProps("film-selected", { data: st.sel }); draw(); return; }
      st.drag = { kind: "ruler" };
      st.t = tc; setProps("film-time", { data: st.t }); draw();
    } else if (kind === "remove") {
      var spec = clone(st.spec); delete spec.lanes[target.getAttribute("data-lane")];
      commit(spec, null);
      return;
    } else if (kind === "key") {
      var lane = target.getAttribute("data-lane"), i = +target.getAttribute("data-i");
      var spec2 = clone(st.spec), keys = keyList(spec2, lane);
      st.sel = { lane: lane, i: i };
      setProps("film-selected", { data: st.sel });
      st.drag = { kind: "key", lane: lane, key: keys[i], spec: spec2, moved: false, startX: e.clientX, startY: e.clientY };
      draw();
    } else if (kind === "band") {
      st.drag = { kind: "ruler" };
      st.t = g.t(x); setProps("film-time", { data: st.t }); draw();
    }
    e.preventDefault();
  }

  function onMove(e) {
    var d = st.drag;
    if (!d) return;
    var el = document.getElementById("film-timeline"), r = el.getBoundingClientRect(), g = geom(el);
    if (d.kind === "ruler") {
      st.t = g.t(e.clientX - r.left); setProps("film-time", { data: st.t }); draw();
      return;
    }
    if (Math.abs(e.clientX - d.startX) + Math.abs(e.clientY - d.startY) < 3 && !d.moved) return;
    d.moved = true;
    d.key.t = Math.round(g.t(e.clientX - r.left) * 100) / 100;
    if (d.lane !== "pictures" && d.lane !== "looks") {
      var li = lanes().indexOf(d.lane), top = laneTop(li), m = (st.meta.lanes || {})[d.lane] || { lo: 0, hi: 1 };
      var f = 1 - (e.clientY - r.top - top - 6) / (LANE - 12);
      var v = m.lo + Math.min(Math.max(f, 0), 1) * (m.hi - m.lo);
      var stp = m.step || 0.01;
      d.key.v = Math.round(v / stp) * stp;
    }
    st.spec = d.spec;
    draw();
  }

  function onUp() {
    var d = st.drag;
    st.drag = null;
    if (!d || d.kind !== "key" || !d.moved) return;
    var spec = d.spec, keys = keyList(spec, d.lane), i = keys.indexOf(d.key);
    st.sel = { lane: d.lane, i: i };
    if (d.lane === "pictures" || d.lane === "looks") { st.t = d.key.t; setProps("film-time", { data: st.t }); }
    commit(spec, st.sel);
  }

  function onDouble(e) {
    var target = e.target, lane = target.getAttribute && target.getAttribute("data-lane");
    if (!st.spec || !lane || lane === "pictures" || lane === "looks" || target.getAttribute("data-kind") !== "band") return;
    var el = document.getElementById("film-timeline"), r = el.getBoundingClientRect(), g = geom(el);
    var t = Math.round(g.t(e.clientX - r.left) * 100) / 100;
    var spec = clone(st.spec), keys = keyList(spec, lane);
    var key = { t: t, v: laneValue(keys, t), ease: "linear" };
    keys.push(key);
    keys = keyList(spec, lane);
    commit(spec, { lane: lane, i: keys.indexOf(key) });
  }

  function onKey(e) {
    var tag = (e.target && e.target.tagName) || "";
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(tag) || (e.target && e.target.isContentEditable)) return;
    var view = document.getElementById("alc-video-view");
    if (!view || view.style.display === "none") return;
    if ((e.key === "Delete" || e.key === "Backspace") && st.sel && st.spec) {
      var spec = clone(st.spec), keys = keyList(spec, st.sel.lane);
      if (keys && keys[st.sel.i]) {
        keys.splice(st.sel.i, 1);
        if (st.sel.lane !== "pictures" && st.sel.lane !== "looks" && !keys.length) delete spec.lanes[st.sel.lane];
        st.sel = null;
        commit(spec, null);
      }
      e.preventDefault();
    } else if (e.key === " ") {
      toggle(); e.preventDefault();
    }
  }

  // ── draft playback ──
  function toggle() {
    st.playing = !st.playing;
    st.wall = performance.now(); st.from = st.t;
    var b = document.getElementById("film-play");
    if (b) b.textContent = st.playing ? "❚❚" : "▶";
    if (st.playing) advance();
  }
  function advance() {
    if (!st.playing || !st.meta) return;
    var L = st.meta.length || 1, t = st.from + (performance.now() - st.wall) / 1000;
    if (t >= L) {
      if (st.spec && st.spec.loop === false) {   // a film that does not loop stops at its end
        st.playing = false;
        var b = document.getElementById("film-play");
        if (b) b.textContent = "▶";
        t = L;
      } else {
        t = t % L;
      }
    }
    st.t = t; setProps("film-time", { data: t }); draw();
  }

  // ── the frame gate ──
  var gate = window.filmGate = {
    out: false, held: false, at: 0,
    ask: function () {
      if (this.out && Date.now() - this.at < 120000) { this.held = true; return window.dash_clientside.no_update; }
      this.out = true; this.held = false; this.at = Date.now();
      return { seq: this.at };
    },
    release: function () {
      this.out = false;
      if (st.playing) { advance(); return; }
      if (this.held) { this.held = false; this.out = true; this.at = Date.now(); setProps("film-request", { data: { seq: this.at } }); }
    }
  };
  new MutationObserver(function (records) {
    records.forEach(function (r) {
      if (r.target.id === "film-frame" && r.target.getAttribute("data-dash-is-loading") !== "true") gate.release();
    });
  }).observe(document.documentElement, { subtree: true, attributes: true, attributeFilter: ["data-dash-is-loading"] });

  // ── pictures dragged from the collection ──
  function draggedPath(e) {
    var types = (e.dataTransfer && e.dataTransfer.types) || [];
    return Array.prototype.indexOf.call(types, "text/arcana-path") >= 0;
  }
  document.addEventListener("dragstart", function (e) {
    var img = e.target.closest && e.target.closest("#film-cast img");
    if (!img) return;
    var path;
    try { path = JSON.parse(img.id).index; } catch (err) { return; }
    e.dataTransfer.setData("text/arcana-path", path);
    e.dataTransfer.effectAllowed = "copy";
  });
  document.addEventListener("dragover", function (e) {
    var el = e.target.closest && e.target.closest("#film-timeline");
    if (!el || !draggedPath(e) || !st.spec) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    var g = geom(el), r = el.getBoundingClientRect();
    st.dropT = Math.round(g.t(e.clientX - r.left) * 20) / 20;
    draw();
  });
  document.addEventListener("dragleave", function (e) {
    if (st.dropT !== null && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest("#film-timeline"))) {
      st.dropT = null; draw();
    }
  });
  document.addEventListener("drop", function (e) {
    var el = e.target.closest && e.target.closest("#film-timeline");
    if (!el || !draggedPath(e) || !st.spec) return;
    e.preventDefault();
    var path = e.dataTransfer.getData("text/arcana-path"), t = st.dropT;
    st.dropT = null;
    if (!path || t === null) { draw(); return; }
    var spec = clone(st.spec), pics = keyList(spec, "pictures");
    while (pics.some(function (p) { return Math.abs(p.t - t) < 0.01; })) t = Math.round((t + 0.05) * 100) / 100;
    var key = { t: t, ref: path, ease: "linear" };
    pics.push(key);
    pics = keyList(spec, "pictures");
    if (spec.length && t >= spec.length) delete spec.length;   // past the end: the length follows the pictures
    st.t = t;
    setProps("film-time", { data: t });
    commit(spec, { lane: "pictures", i: pics.indexOf(key) });
  });

  window.filmTimeline = {
    draw: function (spec, meta, t, sel) {
      if (st.drag) return;                  // a drag draws itself until it ends
      st.spec = spec ? clone(spec) : null;
      st.meta = meta; st.t = +t || 0; st.sel = sel || null;
      if (st.spec) { st.spec.lanes = st.spec.lanes || {}; st.spec.pictures = st.spec.pictures || []; st.spec.looks = st.spec.looks || []; }
      draw();
    }
  };

  document.addEventListener("pointerdown", function (e) {
    if (e.target.closest && e.target.closest("#film-timeline")) onDown(e);
    if (e.target.closest && e.target.closest("#film-play")) { toggle(); e.preventDefault(); }
  });
  document.addEventListener("pointermove", onMove);
  document.addEventListener("pointerup", onUp);
  document.addEventListener("dblclick", function (e) { if (e.target.closest && e.target.closest("#film-timeline")) onDouble(e); });
  document.addEventListener("keydown", onKey);
  window.addEventListener("resize", draw);
})();
