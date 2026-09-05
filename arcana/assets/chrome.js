/* Fold the header band away.
 *
 * The band -- logo, the four mode tabs, the dataset picker -- was 176px tall,
 * because Dash 4 stacks radio options into a column and its own class beats the
 * inline-block asked for in labelStyle. custom.css lays them in a row, which
 * brings it to about 44px; this makes even that foldable, because on a laptop
 * the plot and the results are what the screen is for.
 *
 * Like splitter.js, the control is created here and lives in <body> rather than
 * inside the Dash tree: React owns #main-row and everything under it, and is
 * entitled to replace its children whenever a callback fires.
 */
(function () {
  "use strict";

  var STORE = "arcana.chrome.v1";
  var FOLDED = "arc-chrome-folded";

  function saved() {
    try {
      return window.localStorage.getItem(STORE) === "folded";
    } catch (e) {
      return false;                    // private window, or site data blocked
    }
  }

  function remember(folded) {
    try {
      window.localStorage.setItem(STORE, folded ? "folded" : "open");
    } catch (e) { /* a lost preference is not worth an error */ }
  }

  var btn = document.createElement("button");
  btn.id = "arc-fold";
  btn.type = "button";
  document.body.appendChild(btn);

  function paint() {
    var folded = document.body.classList.contains(FOLDED);
    // Say what the click will DO, not what the state is: a control labelled
    // with its own state reads as a status line and gets ignored.
    btn.textContent = folded ? "▾ show header" : "▴ hide header";
    btn.title = folded ? "Show the header band" : "Hide the header band";
  }

  function apply(folded) {
    document.body.classList.toggle(FOLDED, folded);
    remember(folded);
    paint();
    // The plot sizes itself to its container, and that container just changed
    // height. Without this it keeps the old geometry until the next redraw.
    window.dispatchEvent(new Event("resize"));
  }

  btn.addEventListener("click", function () {
    apply(!document.body.classList.contains(FOLDED));
  });

  // The header row has no id of its own in the layout, so tag the first flex
  // row that holds the mode switcher. Re-tagged on a schedule because Dash may
  // rebuild it.
  function tag() {
    var ms = document.getElementById("mode-select");
    if (!ms) return;
    var row = ms.parentElement;
    if (row && row.id !== "arc-header-row") {
      row.id = "arc-header-row";
    }
  }

  tag();
  apply(saved());
  setInterval(tag, 900);
})();
