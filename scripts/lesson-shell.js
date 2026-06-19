(function () {
  'use strict';

  var TOPIC_BY_BASE = {
    'Backend & System Design': 'backend-design',
    'Gen AI & Agentic AI': 'gen-ai-agentic-ai'
  };

  function decodePath(pathname) {
    try {
      return decodeURIComponent(pathname);
    } catch (e) {
      return pathname;
    }
  }

  function sitePathFromHref(href) {
    var a = document.createElement('a');
    a.href = href;
    var path = decodePath(a.pathname);
    for (var base in TOPIC_BY_BASE) {
      var idx = path.indexOf(base);
      if (idx !== -1) {
        return {
          base: base,
          topicId: TOPIC_BY_BASE[base],
          relative: path.slice(idx + base.length + 1)
        };
      }
    }
    if (/index\.html$/.test(path) || path.endsWith('/')) {
      return { home: true };
    }
    return null;
  }

  function hashForLesson(topicId, relativePath) {
    return topicId + '/' + relativePath.replace(/\.html$/, '');
  }

  function currentTopicIdFromLocation() {
    var path = decodePath(window.location.pathname);
    for (var base in TOPIC_BY_BASE) {
      if (path.indexOf(base) !== -1) {
        return TOPIC_BY_BASE[base];
      }
    }
    return 'backend-design';
  }

  function handleClick(e) {
    if (window.self === window.top) {
      return;
    }
    var link = e.target.closest('a[href]');
    if (!link) {
      return;
    }
    var href = link.getAttribute('href');
    if (
      !href ||
      href.charAt(0) === '#' ||
      href.indexOf('mailto:') === 0 ||
      href.indexOf('http://') === 0 ||
      href.indexOf('https://') === 0
    ) {
      return;
    }

    var resolved = sitePathFromHref(link.href);
    if (!resolved) {
      return;
    }

    e.preventDefault();

    if (resolved.home) {
      window.parent.location.hash = currentTopicIdFromLocation();
      return;
    }

    if (resolved.topicId && resolved.relative) {
      window.parent.location.hash = hashForLesson(resolved.topicId, resolved.relative);
    }
  }

  document.addEventListener('click', handleClick);
})();
