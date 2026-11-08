/* The Archive tab's tree (ui_archive.py): drag a row onto a folder, a kind, or (for a collection)
 * another collection, to move it there. The move itself is the `archive move` op, run by the
 * callback that reads arc-drop. */
(function () {
  "use strict";
  var TYPE = "text/arcana-archive";

  function rowOf(el) { return el && el.closest ? el.closest(".arc-row") : null; }
  function carries(e) {
    return Array.prototype.indexOf.call((e.dataTransfer && e.dataTransfer.types) || [], TYPE) >= 0;
  }
  function clearOver() {
    document.querySelectorAll(".arc-row.over").forEach(function (r) { r.classList.remove("over"); });
  }

  document.addEventListener("dragstart", function (e) {
    var row = rowOf(e.target);
    if (!row || !row.closest("#arc-tree")) return;
    var id = row.getAttribute("data-id");
    if (!id || id.indexOf("kind:") === 0) { e.preventDefault(); return; }
    e.dataTransfer.setData(TYPE, id);
    e.dataTransfer.effectAllowed = "move";
  });
  document.addEventListener("dragover", function (e) {
    var row = rowOf(e.target);
    if (!row || !carries(e) || !row.classList.contains("arc-drop-target")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    clearOver();
    row.classList.add("over");
  });
  document.addEventListener("dragleave", function (e) {
    var row = rowOf(e.target);
    if (row) row.classList.remove("over");
  });
  document.addEventListener("drop", function (e) {
    var row = rowOf(e.target);
    clearOver();
    if (!row || !carries(e) || !row.classList.contains("arc-drop-target")) return;
    e.preventDefault();
    var id = e.dataTransfer.getData(TYPE), into = row.getAttribute("data-id");
    if (id && into && id !== into && window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props("arc-drop", { data: { id: id, into: into, at: Date.now() } });
    }
  });
  document.addEventListener("dragend", clearOver);
})();
