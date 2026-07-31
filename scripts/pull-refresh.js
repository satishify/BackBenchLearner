/**
 * Pull-to-refresh for nested scroll layouts (body overflow:hidden).
 * Native browser PTR does not fire when the document itself cannot overscroll.
 */
(function (global) {
  'use strict';

  var THRESHOLD = 70;
  var HINT_ID = 'bbl-pull-refresh-hint';

  function canUseTouch() {
    return 'ontouchstart' in global || (global.navigator && global.navigator.maxTouchPoints > 0);
  }

  function ensureHint() {
    var el = global.document.getElementById(HINT_ID);
    if (el) return el;
    el = global.document.createElement('div');
    el.id = HINT_ID;
    el.className = 'bbl-pull-refresh-hint';
    el.setAttribute('aria-hidden', 'true');
    el.textContent = 'Pull to refresh';
    global.document.body.appendChild(el);
    return el;
  }

  function setHint(dy, armed) {
    var el = ensureHint();
    var progress = Math.max(0, Math.min(1, dy / THRESHOLD));
    el.style.opacity = String(0.35 + progress * 0.65);
    el.style.transform = 'translate(-50%, ' + Math.min(dy * 0.45, 48) + 'px)';
    el.classList.toggle('ready', dy >= THRESHOLD);
    el.textContent = dy >= THRESHOLD ? 'Release to refresh' : 'Pull to refresh';
    el.classList.toggle('visible', armed && dy > 12);
  }

  function hideHint() {
    var el = global.document.getElementById(HINT_ID);
    if (!el) return;
    el.classList.remove('visible', 'ready');
    el.style.opacity = '0';
    el.style.transform = 'translate(-50%, -8px)';
  }

  /**
   * @param {object} options
   * @param {Element|Document|Window} options.target - element receiving touches
   * @param {function(): number} options.getScrollTop
   * @param {function(): void} options.onRefresh
   * @param {function(): boolean} [options.isBlocked] - return true to ignore gesture
   */
  function attachPullToRefresh(options) {
    if (!canUseTouch() || !options || !options.getScrollTop || !options.onRefresh) return function () {};

    var target = options.target || global.document;
    var startY = 0;
    var lastDy = 0;
    var armed = false;
    var pulling = false;

    function blocked() {
      return typeof options.isBlocked === 'function' && options.isBlocked();
    }

    function onStart(e) {
      if (blocked() || !e.touches || e.touches.length !== 1) {
        armed = false;
        return;
      }
      if (options.getScrollTop() > 1) {
        armed = false;
        return;
      }
      armed = true;
      pulling = false;
      startY = e.touches[0].clientY;
      lastDy = 0;
    }

    function onMove(e) {
      if (!armed || blocked() || !e.touches || e.touches.length !== 1) return;
      if (options.getScrollTop() > 1) {
        armed = false;
        pulling = false;
        hideHint();
        return;
      }
      var dy = e.touches[0].clientY - startY;
      lastDy = dy;
      if (dy < 10) {
        if (pulling) hideHint();
        pulling = false;
        return;
      }
      pulling = true;
      setHint(dy, true);
      if (e.cancelable) e.preventDefault();
    }

    function onEnd() {
      if (pulling && lastDy >= THRESHOLD && !blocked()) {
        hideHint();
        options.onRefresh();
      } else {
        hideHint();
      }
      armed = false;
      pulling = false;
      lastDy = 0;
    }

    target.addEventListener('touchstart', onStart, { passive: true });
    target.addEventListener('touchmove', onMove, { passive: false });
    target.addEventListener('touchend', onEnd, { passive: true });
    target.addEventListener('touchcancel', onEnd, { passive: true });

    return function detach() {
      target.removeEventListener('touchstart', onStart);
      target.removeEventListener('touchmove', onMove);
      target.removeEventListener('touchend', onEnd);
      target.removeEventListener('touchcancel', onEnd);
      hideHint();
    };
  }

  global.BBL = global.BBL || {};
  global.BBL.attachPullToRefresh = attachPullToRefresh;
})(window);
