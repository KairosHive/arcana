// scroll.js -- keep the page where the eye already is.
//
// Dash re-renders a container by replacing its children wholesale, and the
// browser resets scrollTop when it does. That is right in one place and wrong
// in another, so this handles the two cases separately:
//
//   * The collection rail keeps its position. Clicking R or T on a thumbnail
//     rebuilds the whole gallery to move one badge, and the rail jumped back to
//     the top every time -- so choosing a reference from picture forty meant
//     scrolling back down to it before choosing a target.
//
//   * The results column goes to the top. A new search writes a new list, and
//     staying at the old offset drops you into the middle of results you have
//     not seen, with no way to tell you are not at the beginning.
//
// Written as a MutationObserver rather than a Dash clientside callback because
// the position has to be read BEFORE the children change, and a callback only
// runs after. Assets in this folder are loaded automatically by Dash.
(function () {
  "use strict";

  // Restore within the same frame the mutation lands in, so the jump is never
  // painted. requestAnimationFrame would show one frame at the top first.
  function keepPosition(el) {
    var last = el.scrollTop;
    el.addEventListener("scroll", function () {
      // Ignore the reset itself: a re-render reports scrollTop 0 before we put
      // it back, and recording that would defeat the whole thing.
      if (el.scrollTop !== 0 || last === 0) last = el.scrollTop;
    }, { passive: true });

    new MutationObserver(function () {
      if (last && el.scrollTop === 0) el.scrollTop = last;
    }).observe(el, { childList: true, subtree: false });
  }

  // The list itself does not scroll -- the column around it does. Finding that
  // column needs more than a computed overflow-y: the results list is styled
  // overflow-x:hidden, and CSS then computes overflow-y to "auto" even though
  // the element never scrolls. So require that it actually overflows too.
  function scrollerFor(el) {
    var node = el;
    while (node && node !== document.body) {
      var oy = getComputedStyle(node).overflowY;
      if ((oy === "auto" || oy === "scroll") &&
          node.scrollHeight > node.clientHeight + 1) {
        return node;
      }
      node = node.parentElement;
    }
    return el;
  }

  // Called from a Dash clientside callback rather than driven by a
  // MutationObserver here. React reconciles the results list by reusing the
  // DOM nodes it already has and updating them in place, so going from twelve
  // cards to two adds no nodes at all and an addedNodes check never fires.
  // The callback, by contrast, runs exactly when Dash sets new children.
  window.arcanaResultsToTop = function (id) {
    var el = document.getElementById(id);
    if (!el || !el.offsetParent) return;      // not the visible panel
    var scroller = scrollerFor(el);
    if (!scroller || scroller.scrollTop <= 0) return;

    // Animated by hand rather than with scrollTo({behavior:"smooth"}), which
    // is silently a no-op in some embedded browsers -- it left the column
    // exactly where it was while a plain scrollTop assignment worked. Twelve
    // lines of easing is worth not having the behaviour depend on the shell
    // the app happens to be running in.
    // No animation when it cannot be seen: requestAnimationFrame does not run
    // in a hidden tab, so easing there would leave the column exactly where it
    // was and the jump would appear on the way back.
    var reduce = window.matchMedia &&
                 window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || document.visibilityState !== "visible") {
      scroller.scrollTop = 0;
      return;
    }

    var from = scroller.scrollTop;
    var start = null;
    var ms = Math.min(420, 120 + from * 0.25);   // longer drops take longer
    var done = false;
    function step(now) {
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / ms);
      var eased = 1 - Math.pow(1 - t, 3);        // ease-out cubic
      scroller.scrollTop = from * (1 - eased);
      if (t < 1) requestAnimationFrame(step);
      else done = true;
    }
    requestAnimationFrame(step);

    // requestAnimationFrame is starved whenever the window is not painting --
    // behind another window, minimised, an inactive tab -- and visibilityState
    // does not always say so. Without this the list would simply stay where it
    // was in those cases, which is the bug we are fixing. Land it anyway.
    setTimeout(function () {
      if (!done && scroller.scrollTop > 1) scroller.scrollTop = 0;
    }, ms + 120);
  };

  var wanted = {
    "moodboard-gallery": keepPosition,
  };
  var attached = {};

  // The layout arrives after this file does, and the panels mount and unmount
  // as tabs change, so watch for them rather than assuming they are there.
  function attach() {
    Object.keys(wanted).forEach(function (id) {
      if (attached[id]) return;
      var el = document.getElementById(id);
      if (!el) return;
      wanted[id](el);
      attached[id] = true;
    });
  }

  if (document.readyState !== "loading") attach();
  document.addEventListener("DOMContentLoaded", attach);
  new MutationObserver(attach).observe(document.documentElement,
                                       { childList: true, subtree: true });
})();
