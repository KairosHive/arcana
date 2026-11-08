/* The collection viewer's keys (ui_collection.py): the left and right arrows
 * step through the active collection, Escape closes. They act only while the
 * viewer is open and never while a text field has the focus. */
(function () {
  "use strict";
  var KEYS = { ArrowLeft: "collection-view-prev", ArrowRight: "collection-view-next",
               Escape: "collection-view-close" };
  document.addEventListener("keydown", function (e) {
    var viewer = document.getElementById("collection-viewer");
    if (!viewer || viewer.style.display === "none" || !KEYS[e.key]) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    var button = document.getElementById(KEYS[e.key]);
    if (button) { button.click(); e.preventDefault(); }
  });
})();
