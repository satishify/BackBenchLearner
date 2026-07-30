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
  // Reading progress bar
  // ----------------------------------------------------------------------

  function initReadProgress() {
    var bar = doc.getElementById('read-progress-bar');
    if (!bar) return;
    var pending = false;

    function paint() {
      pending = false;
      var scrollable = doc.documentElement.scrollHeight - global.innerHeight;
      var ratio = scrollable > 0 ? global.scrollY / scrollable : 1;
      bar.style.width = Math.max(0, Math.min(1, ratio)) * 100 + '%';
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
    paint();
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

    if (!Progress || !Nav || !topicId || !lessonHash) return;
    var location = Nav.locate(topicId, lessonHash);
    if (!location) return;

    Progress.markVisited(Nav.lessonId(topicId, lessonHash));
    renderMeta(location);
    renderActions(location);
    initReadProgress();

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
