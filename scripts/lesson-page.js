/**
 * Runs inside every lesson page (standalone or in the shell's iframe).
 *
 * Responsibilities:
 *   - fill in the meta line under the title (chapter, position, read time)
 *   - drive the scroll-linked reading progress bar
 *   - render the Mark as complete toggle and prev/next navigation
 *   - keep in-frame link clicks routed through the shell's hash
 *   - wire up the copy button on code blocks
 */
(function (global) {
  'use strict';

  var doc = global.document;
  var BBL = global.BBL || {};
  var Progress = BBL.Progress;
  var Nav = BBL.Nav;

  var topicId = doc.body.getAttribute('data-topic');
  var lessonHash = doc.body.getAttribute('data-lesson');
  var inFrame = global.self !== global.top;

  // ----------------------------------------------------------------------
  // Reading progress bar + on-this-page TOC
  // ----------------------------------------------------------------------

  function scrollRatio() {
    var scrollable = doc.documentElement.scrollHeight - global.innerHeight;
    if (scrollable <= 0) return 1;
    return Math.max(0, Math.min(1, global.scrollY / scrollable));
  }

  function postToShell(type, extra) {
    if (!inFrame || !global.parent || global.parent === global) return;
    var active = null;
    if (sectionIndex.length) {
      var offset = 110;
      active = sectionIndex[0];
      for (var i = 0; i < sectionIndex.length; i += 1) {
        var el = doc.getElementById(sectionIndex[i].id);
        if (!el) continue;
        if (el.getBoundingClientRect().top <= offset) active = sectionIndex[i];
      }
    }
    var payload = {
      type: type,
      progress: scrollRatio(),
      activeId: active ? active.id : '',
      sections: sectionIndex.map(function (section) {
        return { id: section.id, n: section.n, title: section.title };
      })
    };
    if (extra) {
      for (var key in extra) {
        if (Object.prototype.hasOwnProperty.call(extra, key)) payload[key] = extra[key];
      }
    }
    try {
      global.parent.postMessage(payload, '*');
    } catch (e) { /* ignore */ }
  }

  function initReadProgress() {
    var bar = doc.getElementById('read-progress-bar');
    var fill = doc.getElementById('rp-bar-fill');
    var pctEl = doc.getElementById('rp-pct');
    var pending = false;

    function paint() {
      pending = false;
      var ratio = scrollRatio();
      var pct = Math.round(ratio * 100);
      if (bar) bar.style.width = pct + '%';
      if (fill) fill.style.width = pct + '%';
      if (pctEl) pctEl.textContent = pct + '%';
      paintActiveSection();
      postToShell('bbl-lesson-progress');
      if (ratio > 0.9) markNearlyRead();
    }

    global.addEventListener(
      'scroll',
      function () {
        if (pending) return;
        pending = true;
        global.requestAnimationFrame(paint);
      },
      { passive: true }
    );
    global.addEventListener('resize', paint, { passive: true });
    paint();
  }

  function initShellBridge() {
    if (!inFrame) return;
    // Prefer the shell rail; keep in-page rail as a fallback for narrow shells.
    global.addEventListener('message', function (event) {
      var data = event.data;
      if (!data || data.type !== 'bbl-scroll-to' || !data.id) return;
      var target = doc.getElementById(data.id);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  var sectionIndex = [];

  function collectSectionHeadings() {
    // Prefer lesson body headings. Do not require them to stay inside .wrapper —
    // broken markup (e.g. mermaid) can move nodes during HTML repair.
    var nodes = doc.querySelectorAll('h2');
    var out = [];
    for (var i = 0; i < nodes.length; i += 1) {
      var heading = nodes[i];
      if (heading.closest('.lesson-rail, .on-this-page, .read-progress')) continue;
      out.push(heading);
    }
    return out;
  }

  function ensureProgressCard() {
    var card = doc.querySelector('.reading-progress-card');
    if (card) return card;
    var rail = doc.querySelector('.lesson-rail');
    if (!rail) {
      var layout = doc.querySelector('.lesson-layout') || doc.body;
      rail = doc.createElement('aside');
      rail.className = 'lesson-rail';
      rail.setAttribute('aria-label', 'Reading progress');
      layout.appendChild(rail);
    }
    card = doc.createElement('div');
    card.className = 'reading-progress-card';
    card.innerHTML =
      '<div class="reading-progress-head">' +
      '<span class="reading-progress-label">Reading progress</span>' +
      '<span class="reading-progress-pct" id="rp-pct">0%</span>' +
      '</div>' +
      '<div class="reading-progress-track" aria-hidden="true"><span id="rp-bar-fill"></span></div>';
    rail.appendChild(card);
    return card;
  }

  /** Number h2 sections and build the full "On this page" list. */
  function initSectionNav() {
    var card = ensureProgressCard();
    if (!card) return;

    var legacy = doc.getElementById('rp-section');
    if (legacy) legacy.remove();

    var headings = collectSectionHeadings();
    sectionIndex = [];

    var nav = card.querySelector('.on-this-page');
    if (!nav) {
      nav = doc.createElement('nav');
      nav.className = 'on-this-page';
      nav.setAttribute('aria-label', 'On this page');
      card.appendChild(nav);
    }
    nav.hidden = false;
    nav.style.display = 'block';

    if (!headings.length) {
      nav.innerHTML =
        '<div class="on-this-page-label">On this page</div>' +
        '<p class="on-this-page-empty">No section headings on this page.</p>';
      return;
    }

    var html = '<div class="on-this-page-label">On this page</div><ol class="on-this-page-list" id="otp-list">';
    headings.forEach(function (heading, index) {
      var n = index + 1;
      if (!heading.id) heading.id = 'section-' + n;
      var label = heading.textContent.replace(/^\d+\.\s*/, '').trim() || 'Section ' + n;
      if (!heading.querySelector('.section-num')) {
        heading.textContent = '';
        var num = doc.createElement('span');
        num.className = 'section-num';
        num.textContent = n + '.';
        heading.appendChild(num);
        heading.appendChild(doc.createTextNode(' ' + label));
      } else {
        label = heading.textContent.replace(/^\d+\.\s*/, '').trim() || label;
      }
      sectionIndex.push({ id: heading.id, n: n, title: label });
      html +=
        '<li><a href="#' +
        heading.id +
        '" data-section-id="' +
        heading.id +
        '"><span class="otp-num">' +
        n +
        '.</span>' +
        escapeHtml(label) +
        '</a></li>';
    });
    html += '</ol>';
    nav.innerHTML = html;

    var list = doc.getElementById('otp-list');
    if (list && list.getAttribute('data-bound') !== '1') {
      list.setAttribute('data-bound', '1');
      list.addEventListener('click', function (event) {
        var anchor = event.target.closest('a[href^="#"]');
        if (!anchor) return;
        var id = anchor.getAttribute('href').slice(1);
        var target = doc.getElementById(id);
        if (!target) return;
        event.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }

    postToShell('bbl-lesson-outline');
  }

  function paintActiveSection() {
    if (!sectionIndex.length) return;

    var offset = 110;
    var active = sectionIndex[0];
    for (var i = 0; i < sectionIndex.length; i += 1) {
      var el = doc.getElementById(sectionIndex[i].id);
      if (!el) continue;
      if (el.getBoundingClientRect().top <= offset) active = sectionIndex[i];
    }

    var list = doc.getElementById('otp-list');
    if (!list) return;
    var links = list.querySelectorAll('a[data-section-id]');
    for (var j = 0; j < links.length; j += 1) {
      var on = links[j].getAttribute('data-section-id') === active.id;
      links[j].classList.toggle('active', on);
      if (on) links[j].setAttribute('aria-current', 'true');
      else links[j].removeAttribute('aria-current');
    }
  }

  var nudged = false;

  /** Past ~90% of the page, draw attention to the button without clicking it. */
  function markNearlyRead() {
    if (nudged) return;
    nudged = true;
    var button = doc.querySelector('.lesson-complete');
    if (button && !button.classList.contains('is-done')) button.classList.add('is-ready');
  }

  // ----------------------------------------------------------------------
  // Meta line and actions
  // ----------------------------------------------------------------------

  function link(node) {
    if (node.type === 'quiz') {
      return {
        href: relRoot() + '/quiz.html?t=' + encodeURIComponent(topicId) + '&c=' + encodeURIComponent(node.chapter.id),
        label: 'Chapter quiz'
      };
    }
    var topic = Nav.topic(topicId);
    var code = node.code || Nav.lessonCode(topicId, node.chapter, node.indexInChapter);
    return {
      href: relRoot() + '/' + encodePath(topic.basePath + '/' + node.lesson.path),
      label: code + ' ' + node.lesson.label
    };
  }

  function encodePath(path) {
    return path.split('/').map(encodeURIComponent).join('/');
  }

  function relRoot() {
    var depth = lessonHash.split('/').length;
    var parts = [];
    for (var i = 0; i < depth; i += 1) parts.push('..');
    return parts.join('/');
  }

  function moduleLabel(chapter) {
    var module = chapter.module ? Nav.module(topicId, chapter.module) : null;
    return module ? module.title : null;
  }

  function renderMeta(location) {
    var host = doc.getElementById('lesson-meta');
    if (!host) return;
    var bits = [];
    var module = moduleLabel(location.chapter);
    if (module) bits.push(escapeHtml(module));
    bits.push(escapeHtml(location.chapter.title));
    bits.push(escapeHtml(location.code || Nav.lessonCode(topicId, location.chapter, location.indexInChapter)));
    bits.push('Lesson ' + location.indexInChapter + ' of ' + location.chapterTotal);
    bits.push(location.lesson.minutes + ' min read');
    host.innerHTML = bits
      .map(function (bit) {
        return '<span>' + bit + '</span>';
      })
      .join('<span class="meta-sep">&middot;</span>');
  }

  function renderActions(location) {
    var host = doc.getElementById('lesson-actions');
    if (!host) return;
    var id = Nav.lessonId(topicId, lessonHash);
    var done = Progress.isDone(id);

    var html = '<button type="button" class="lesson-complete' + (done ? ' is-done' : '') + '">';
    html += '<span class="tick" aria-hidden="true">' + (done ? '&#10003;' : '') + '</span>';
    html += '<span class="label">' + (done ? 'Completed' : 'Mark as complete') + '</span>';
    html += '</button>';

    html += '<div class="lesson-nav">';
    if (location.prev) {
      var prev = link(location.prev);
      html += '<a class="lesson-nav-prev" href="' + prev.href + '"><span>Previous</span>' + escapeHtml(prev.label) + '</a>';
    } else {
      html += '<span class="lesson-nav-spacer"></span>';
    }
    if (location.next) {
      var next = link(location.next);
      var isQuiz = location.next.type === 'quiz';
      html +=
        '<a class="lesson-nav-next' +
        (isQuiz ? ' is-quiz' : '') +
        '" href="' +
        next.href +
        '"><span>' +
        (isQuiz ? 'End of chapter' : 'Next') +
        '</span>' +
        escapeHtml(isQuiz ? 'Take the chapter quiz' : next.label) +
        '</a>';
    } else {
      html += '<span class="lesson-nav-spacer"></span>';
    }
    html += '</div>';

    host.innerHTML = html;

    host.querySelector('.lesson-complete').addEventListener('click', function () {
      var nowDone = Progress.toggleDone(id);
      paintCompleteButton(nowDone);
    });
  }

  function paintCompleteButton(done) {
    var button = doc.querySelector('.lesson-complete');
    if (!button) return;
    button.classList.toggle('is-done', done);
    if (done) button.classList.remove('is-ready');
    button.querySelector('.tick').innerHTML = done ? '&#10003;' : '';
    button.querySelector('.label').textContent = done ? 'Completed' : 'Mark as complete';
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, function (char) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[char];
    });
  }

  // ----------------------------------------------------------------------
  // Link routing: inside the shell, navigation happens via the parent hash
  // ----------------------------------------------------------------------

  function initLinkRouting() {
    if (!inFrame) return;
    doc.addEventListener('click', function (event) {
      var anchor = event.target.closest && event.target.closest('a[href]');
      if (!anchor) return;
      var href = anchor.getAttribute('href');
      if (!href || href.charAt(0) === '#' || /^(https?:|mailto:)/.test(href)) return;

      var resolved = resolveSiteTarget(anchor.href);
      if (!resolved) return;
      event.preventDefault();
      global.parent.location.hash = resolved;
    });
  }

  /** Turn an absolute in-site URL into the shell hash that displays it. */
  function resolveSiteTarget(href) {
    var path;
    try {
      path = decodeURIComponent(new URL(href, global.location.href).pathname);
    } catch (e) {
      return null;
    }
    var search = '';
    try {
      search = new URL(href, global.location.href).search;
    } catch (e) {
      search = '';
    }

    if (/\/quiz\.html$/.test(path)) {
      var params = new URLSearchParams(search);
      var quizTopic = params.get('t');
      var quizChapter = params.get('c');
      if (quizTopic && quizChapter) return quizTopic + '/quiz/' + quizChapter;
      return null;
    }

    var ids = Nav.topicIds();
    for (var i = 0; i < ids.length; i += 1) {
      var topic = Nav.topic(ids[i]);
      var marker = '/' + topic.basePath + '/';
      var at = path.indexOf(marker);
      if (at !== -1) {
        var rest = path.slice(at + marker.length).replace(/\.html$/, '');
        return ids[i] + '/' + rest;
      }
    }
    if (/index\.html$/.test(path) || path.slice(-1) === '/') return topicId;
    return null;
  }

  // ----------------------------------------------------------------------
  // Code copy buttons
  // ----------------------------------------------------------------------

  function initCodeCopy() {
    doc.querySelectorAll('.code-block').forEach(function (block) {
      var button = block.querySelector('.code-copy');
      var code = block.querySelector('code');
      if (!button || !code) return;
      button.addEventListener('click', function () {
        var text = code.textContent;
        var done = function () {
          button.textContent = 'Copied';
          global.setTimeout(function () {
            button.textContent = 'Copy';
          }, 1500);
        };
        if (global.navigator.clipboard) {
          global.navigator.clipboard.writeText(text).then(done, fallbackCopy);
        } else {
          fallbackCopy();
        }

        function fallbackCopy() {
          var area = doc.createElement('textarea');
          area.value = text;
          area.setAttribute('readonly', '');
          area.style.position = 'absolute';
          area.style.left = '-9999px';
          doc.body.appendChild(area);
          area.select();
          try {
            doc.execCommand('copy');
            done();
          } catch (e) {
            button.textContent = 'Press Cmd+C';
          }
          doc.body.removeChild(area);
        }
      });
    });
  }

  // ----------------------------------------------------------------------
  // Boot
  // ----------------------------------------------------------------------

  function boot() {
    initCodeCopy();
    initLinkRouting();
    initShellBridge();
    // TOC + scroll % must not depend on curriculum lookup succeeding.
    initSectionNav();
    initReadProgress();
    postToShell('bbl-lesson-outline');

    if (!Progress || !Nav || !topicId || !lessonHash) return;
    var location = Nav.locate(topicId, lessonHash);
    if (!location) return;

    Progress.markVisited(Nav.lessonId(topicId, lessonHash));
    renderMeta(location);
    renderActions(location);

    Progress.subscribe(function () {
      paintCompleteButton(Progress.isDone(Nav.lessonId(topicId, lessonHash)));
    });
  }

  if (doc.readyState === 'loading') {
    doc.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})(window);
