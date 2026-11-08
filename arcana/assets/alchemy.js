/* The Alchemy stage's before/after line (ui_alchemy.py).
 *
 * Dragging anywhere on the picture moves the line. The position lives in the
 * CSS variable --cut on the .alc-compare box, which Dash never writes (the box
 * has no style prop), so a new preview keeps the line where it was. Nothing
 * here goes through a callback, so the line follows the pointer at frame rate.
 */
(function () {
  "use strict";

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
})();
