/**
 * BackbenchLearner shell: topic tabs, sidebar, iframe routing, progress dashboard.
 * Depends on scripts/curriculum.js and scripts/progress.js.
 */
(function () {
  'use strict';

  var CURRENT_TOPIC_KEY = 'bbl.current-topic';
  var BBL = window.BBL;
  var Nav = BBL.Nav;
  var Progress = BBL.Progress;

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, function (char) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[char];
    });
  }

  function encodePath(path) {
    return path.split('/').map(encodeURIComponent).join('/');
  }

  function getCurrentTopicId() {
    try {
      var id = localStorage.getItem(CURRENT_TOPIC_KEY);
      if (id && Nav.topic(id)) return id;
    } catch (e) { /* ignore */ }
    return Nav.topicIds()[0];
  }

  function setCurrentTopicId(id) {
    try {
      localStorage.setItem(CURRENT_TOPIC_KEY, id);
    } catch (e) { /* ignore */ }
  }

  /**
   * Hash shapes:
   *   #backend-design
   *   #backend-design/backend/what-is-api
   *   #backend-design/quiz/backend
   *   #practice
   *   #practice/backend-design
   *   #cheatsheet
   *   #cheatsheet/genai/module-1-foundations
   */
  function parseHash() {
    var raw = window.location.hash.slice(1);
    var h = raw;
    try {
      h = decodeURIComponent(raw);
    } catch (e) { /* keep raw */ }

    if (!h) return { topicId: getCurrentTopicId(), kind: 'welcome' };

    var parts = h.split('/');

    if (parts[0] === 'practice') {
      if (parts.length === 1) {
        return { topicId: getCurrentTopicId(), kind: 'practice-hub', hash: h };
      }
      var practiceTopic = parts[1];
      if (practiceTopic === 'gen-ai-agentic-ai') practiceTopic = 'genai';
      if (!Nav.topic(practiceTopic)) {
        return { topicId: getCurrentTopicId(), kind: 'practice-hub', hash: h };
      }
      // #practice/genai/module-1 → authored module mock bank
      var mockKey = parts.length >= 3 ? parts.slice(2).join('/') : '';
      return {
        topicId: practiceTopic,
        kind: 'practice-mock',
        mockKey: mockKey,
        hash: h
      };
    }

    if (parts[0] === 'cheatsheet') {
      if (parts.length === 1) {
        return { topicId: getCurrentTopicId(), kind: 'cheatsheet-hub', hash: h };
      }
      var cheatTopic = parts[1];
      if (cheatTopic === 'gen-ai-agentic-ai') cheatTopic = 'genai';
      if (!Nav.topic(cheatTopic)) {
        return { topicId: getCurrentTopicId(), kind: 'cheatsheet-hub', hash: h };
      }
      var sheetSlug = parts.length >= 3 ? parts.slice(2).join('/') : '';
      if (!sheetSlug) {
        return { topicId: cheatTopic, kind: 'cheatsheet-hub', hash: h };
      }
      return {
        topicId: cheatTopic,
        kind: 'cheatsheet',
        sheetSlug: sheetSlug,
        hash: h
      };
    }

    var topicId = parts[0];
    if (!Nav.topic(topicId)) {
      if (topicId === 'gen-ai-agentic-ai') topicId = 'genai';
      if (!Nav.topic(topicId)) return { topicId: getCurrentTopicId(), kind: 'welcome' };
    }

    if (parts.length === 1) return { topicId: topicId, kind: 'welcome' };

    if (parts[1] === 'quiz' && parts.length >= 3) {
      return {
        topicId: topicId,
        kind: 'quiz',
        chapterId: parts.slice(2).join('/'),
        hash: h
      };
    }

    return {
      topicId: topicId,
      kind: 'lesson',
      lessonHash: parts.slice(1).join('/'),
      hash: h
    };
  }

  function lessonUrl(topicId, lesson) {
    var topic = Nav.topic(topicId);
    return encodePath(topic.basePath + '/' + lesson.path);
  }

  function quizUrl(topicId, chapterId) {
    return 'quiz.html?t=' + encodeURIComponent(topicId) + '&c=' + encodeURIComponent(chapterId);
  }

  function practiceUrl(topicId, mockKey) {
    var url = 'practice.html?t=' + encodeURIComponent(topicId);
    if (mockKey) url += '&m=' + encodeURIComponent(mockKey);
    return url;
  }

  function listCheatSheets(topicId) {
    var registry = (BBL.CHEATSHEETS && BBL.CHEATSHEETS[topicId]) || [];
    return registry.slice();
  }

  function findCheatSheet(topicId, slug) {
    var sheets = listCheatSheets(topicId);
    for (var i = 0; i < sheets.length; i += 1) {
      if (sheets[i].slug === slug) return sheets[i];
    }
    return null;
  }

  function cheatsheetUrl(sheet) {
    return encodePath(sheet.path);
  }

  function lessonStage() {
    return document.getElementById('lesson-stage');
  }

  function shellRail() {
    return document.getElementById('shell-lesson-rail');
  }

  function showLessonStage(show) {
    var stage = lessonStage();
    var frame = document.getElementById('content-frame');
    if (stage) stage.classList.toggle('visible', !!show);
    if (frame) frame.classList.toggle('visible', !!show);
    document.body.classList.toggle('is-reading', !!show);
    if (!show) clearShellLessonRail();
  }

  function clearShellLessonRail() {
    var rail = shellRail();
    if (rail) rail.classList.remove('visible');
    var list = document.getElementById('shell-otp-list');
    if (list) list.innerHTML = '';
    var empty = document.getElementById('shell-otp-empty');
    if (empty) empty.remove();
    var pct = document.getElementById('shell-rp-pct');
    if (pct) pct.textContent = '0%';
    var fill = document.getElementById('shell-rp-bar-fill');
    if (fill) fill.style.width = '0%';
  }

  function renderShellLessonRail(payload) {
    var rail = shellRail();
    var list = document.getElementById('shell-otp-list');
    var pct = document.getElementById('shell-rp-pct');
    var fill = document.getElementById('shell-rp-bar-fill');
    if (!rail || !list) return;

    rail.classList.add('visible');
    if (typeof payload.progress === 'number') {
      var p = Math.max(0, Math.min(100, Math.round(payload.progress * 100)));
      if (pct) pct.textContent = p + '%';
      if (fill) fill.style.width = p + '%';
    }

    if (payload.replaceSections) {
      var sections = payload.sections || [];
      if (!sections.length) {
        list.innerHTML = '';
        list.insertAdjacentHTML(
          'afterend',
          '<p class="on-this-page-empty" id="shell-otp-empty">No section headings on this page.</p>'
        );
      } else {
        var empty = document.getElementById('shell-otp-empty');
        if (empty) empty.remove();
        var html = '';
        sections.forEach(function (section) {
          html +=
            '<li><a href="#" data-section-id="' +
            escapeHtml(section.id) +
            '"><span class="otp-num">' +
            section.n +
            '.</span>' +
            escapeHtml(section.title) +
            '</a></li>';
        });
        list.innerHTML = html;
      }
    }

    var activeId = payload.activeId || '';
    var links = list.querySelectorAll('a[data-section-id]');
    for (var i = 0; i < links.length; i += 1) {
      links[i].classList.toggle('active', links[i].getAttribute('data-section-id') === activeId);
    }
  }

  function bindShellLessonRail() {
    var rail = shellRail();
    if (!rail || rail.getAttribute('data-bound') === '1') return;
    rail.setAttribute('data-bound', '1');
    rail.addEventListener('click', function (event) {
      var anchor = event.target.closest('a[data-section-id]');
      if (!anchor) return;
      event.preventDefault();
      var id = anchor.getAttribute('data-section-id');
      var frame = document.getElementById('content-frame');
      if (frame && frame.contentWindow) {
        frame.contentWindow.postMessage({ type: 'bbl-scroll-to', id: id }, '*');
      }
    });

    window.addEventListener('message', function (event) {
      var data = event.data;
      if (!data || typeof data !== 'object') return;
      if (data.type === 'bbl-lesson-outline') {
        renderShellLessonRail({
          sections: data.sections || [],
          progress: data.progress,
          activeId: data.activeId,
          replaceSections: true
        });
      } else if (data.type === 'bbl-lesson-progress') {
        renderShellLessonRail({
          sections: data.sections,
          progress: data.progress,
          activeId: data.activeId,
          replaceSections: false
        });
      }
    });
  }

  function listModuleMocks(topicId) {
    var quizzes = BBL.QUIZZES || {};
    var prefix = topicId + '/mock/';
    return Object.keys(quizzes)
      .filter(function (id) {
        return id.indexOf(prefix) === 0 && quizzes[id].kind === 'mock';
      })
      .sort()
      .map(function (id) {
        return {
          id: id,
          key: id.slice(prefix.length),
          title: quizzes[id].title || id,
          minutes: quizzes[id].minutes || 90,
          count: (quizzes[id].questions || []).length
        };
      });
  }

  function topicQuestionCount(topicId) {
    var quizzes = (BBL.QUIZZES) || {};
    var total = 0;
    Object.keys(quizzes).forEach(function (id) {
      if (id.indexOf(topicId + '/') !== 0 || !quizzes[id].questions) return;
      if (quizzes[id].kind === 'mock') return;
      total += quizzes[id].questions.length;
    });
    return total;
  }

  function topicIcon(topicId) {
    if (topicId === 'genai') {
      return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z"/><path d="M19 14l.8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8L19 14z"/></svg>';
    }
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></svg>';
  }

  function setSidebarOpen(open) {
    var sidebar = document.getElementById('sidebar');
    var backdrop = document.getElementById('sidebar-backdrop');
    var toggle = document.getElementById('sidebar-toggle');
    if (!sidebar || !backdrop || !toggle) return;
    sidebar.classList.toggle('open', open);
    backdrop.classList.toggle('visible', open);
    backdrop.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close lessons menu' : 'Open lessons menu');
  }

  function closeMenus() {
    document.querySelectorAll('.nav-item.open').forEach(function (item) {
      item.classList.remove('open');
      var trigger = item.querySelector('.nav-trigger');
      if (trigger) trigger.setAttribute('aria-expanded', 'false');
    });
    var navBackdrop = document.getElementById('nav-backdrop');
    if (navBackdrop) {
      navBackdrop.classList.remove('visible');
      navBackdrop.hidden = true;
    }
  }

  function openMenu(itemId) {
    closeMenus();
    var item = document.getElementById(itemId);
    if (!item) return;
    item.classList.add('open');
    var trigger = item.querySelector('.nav-trigger');
    if (trigger) trigger.setAttribute('aria-expanded', 'true');
    var navBackdrop = document.getElementById('nav-backdrop');
    if (navBackdrop) {
      navBackdrop.classList.add('visible');
      navBackdrop.hidden = false;
    }
  }

  function renderMainNav(activeTopicId, mode) {
    var learnMenu = document.getElementById('learn-menu');
    var practiceMenu = document.getElementById('practice-menu');
    var cheatMenu = document.getElementById('cheatsheet-menu');
    if (!learnMenu || !practiceMenu) return;

    var learnHtml = '<div class="mega-label">Study topics</div><ul class="mega-list">';
    Nav.topics().forEach(function (topic) {
      var stats = Nav.topicStats(topic.id);
      var active = mode === 'learn' && topic.id === activeTopicId ? ' active' : '';
      learnHtml +=
        '<li><a href="#' +
        topic.id +
        '" data-topic-id="' +
        topic.id +
        '" class="' +
        active.trim() +
        '">';
      learnHtml += '<span class="mega-item-icon" aria-hidden="true">' + topicIcon(topic.id) + '</span>';
      learnHtml += '<span><span class="mega-title">' + escapeHtml(topic.navLabel) + '</span>';
      learnHtml +=
        '<span class="mega-desc">' +
        stats.total +
        ' lessons · ' +
        Nav.formatMinutes(stats.minutes) +
        (stats.percent ? ' · ' + stats.percent + '% done' : '') +
        '</span></span></a></li>';
    });
    learnHtml += '</ul>';
    learnHtml +=
      '<div class="mega-footer"><a href="#' +
      (activeTopicId || Nav.topicIds()[0]) +
      '">Continue learning →</a></div>';
    learnMenu.innerHTML = learnHtml;

    var practiceHtml = '';
    Nav.topics().forEach(function (topic) {
      var mocks = listModuleMocks(topic.id);
      practiceHtml +=
        '<div class="mega-label">' + escapeHtml(topic.navLabel) + '</div>';
      practiceHtml += '<ul class="mega-list">';
      if (mocks.length) {
        mocks.forEach(function (mock) {
          var score = Progress.quizScore(mock.id);
          var active =
            mode === 'practice' && topic.id === activeTopicId ? ' active' : '';
          var shortTitle = mock.title
            .replace(topic.navLabel, '')
            .replace(/^[\s·\-–—]+/, '')
            .trim();
          if (!shortTitle) shortTitle = mock.title;
          var desc = mock.count + ' Q · ' + mock.minutes + ' min';
          if (score) desc += ' · best ' + score.best + '/' + score.total;
          practiceHtml +=
            '<li><a href="#practice/' +
            topic.id +
            '/' +
            mock.key +
            '" class="' +
            active.trim() +
            '">';
          practiceHtml +=
            '<span class="mega-item-icon" aria-hidden="true">' +
            topicIcon(topic.id) +
            '</span>';
          practiceHtml +=
            '<span><span class="mega-title">' +
            escapeHtml(shortTitle) +
            '</span>';
          practiceHtml +=
            '<span class="mega-desc">' + escapeHtml(desc) + '</span></span></a></li>';
        });
      } else {
        var qCount = topicQuestionCount(topic.id);
        var legacy = Progress.quizScore(topic.id + '/mock');
        var desc =
          qCount > 0
            ? qCount + ' questions from chapter quizzes'
            : 'Mocks unlock when chapter quizzes exist';
        if (legacy) desc += ' · best ' + legacy.best + '/' + legacy.total;
        practiceHtml +=
          '<li><a href="#practice/' +
          topic.id +
          '"><span class="mega-item-icon" aria-hidden="true">' +
          topicIcon(topic.id) +
          '</span><span><span class="mega-title">Full-topic mock</span><span class="mega-desc">' +
          escapeHtml(desc) +
          '</span></span></a></li>';
      }
      practiceHtml += '</ul>';
    });
    practiceHtml +=
      '<div class="mega-footer"><a href="#practice">View all practice →</a></div>';
    practiceMenu.innerHTML = practiceHtml;

    if (cheatMenu) {
      var cheatHtml = '';
      Nav.topics().forEach(function (topic) {
        var sheets = listCheatSheets(topic.id);
        if (!sheets.length) return;
        cheatHtml +=
          '<div class="mega-label">' + escapeHtml(topic.navLabel) + '</div>';
        cheatHtml += '<ul class="mega-list">';
        var openSlug = '';
        if (mode === 'cheatsheet') {
          var ch = parseHash();
          if (ch.kind === 'cheatsheet' && ch.topicId === topic.id) openSlug = ch.sheetSlug || '';
        }
        sheets.forEach(function (sheet) {
          var active =
            mode === 'cheatsheet' && sheet.slug === openSlug ? ' active' : '';
          cheatHtml +=
            '<li><a href="#cheatsheet/' +
            topic.id +
            '/' +
            sheet.slug +
            '" class="' +
            active.trim() +
            '">';
          cheatHtml +=
            '<span class="mega-item-icon" aria-hidden="true">' +
            topicIcon(topic.id) +
            '</span>';
          cheatHtml +=
            '<span><span class="mega-title">' +
            escapeHtml(sheet.title) +
            '</span>';
          cheatHtml +=
            '<span class="mega-desc">~' +
            sheet.minutes +
            ' min revision' +
            (sheet.module ? ' · ' + escapeHtml(sheet.module) : '') +
            '</span></span></a></li>';
        });
        cheatHtml += '</ul>';
      });
      if (!cheatHtml) {
        cheatHtml =
          '<div class="mega-label">Revision guides</div><ul class="mega-list"><li><span class="mega-desc">Cheat sheets coming soon</span></li></ul>';
      }
      cheatHtml +=
        '<div class="mega-footer"><a href="#cheatsheet">View all cheat sheets →</a></div>';
      cheatMenu.innerHTML = cheatHtml;
    }

    var learnTrigger = document.getElementById('learn-trigger');
    var practiceTrigger = document.getElementById('practice-trigger');
    var cheatTrigger = document.getElementById('cheatsheet-trigger');
    if (learnTrigger) learnTrigger.classList.toggle('active', mode === 'learn');
    if (practiceTrigger) practiceTrigger.classList.toggle('active', mode === 'practice');
    if (cheatTrigger) cheatTrigger.classList.toggle('active', mode === 'cheatsheet');
  }

  function setHeaderActive(topicId, mode) {
    renderMainNav(topicId, mode || 'learn');
  }

  function progressBar(percent) {
    return (
      '<span class="progress-bar" aria-hidden="true"><span class="progress-bar-fill" style="width:' +
      percent +
      '%"></span></span>'
    );
  }

  function renderSidebar(topicId) {
    var topic = Nav.topic(topicId);
    var sidebar = document.getElementById('sidebar');
    if (!topic || !sidebar) return;

    var site = Nav.siteStats();
    var topicStats = Nav.topicStats(topicId);
    var html = '';

    html += '<div class="sidebar-summary">';
    html +=
      '<div class="sidebar-summary-pct">' +
      topicStats.percent +
      '% <span>complete</span></div>';
    html +=
      '<div class="sidebar-summary-meta">' +
      topicStats.done +
      ' / ' +
      topicStats.total +
      ' lessons';
    if (topicStats.minutesLeft > 0) {
      html += ' · ' + Nav.formatMinutes(topicStats.minutesLeft) + ' left';
    }
    html += '</div>';
    html += progressBar(topicStats.percent);
    if (site.total !== topicStats.total) {
      html +=
        '<div class="sidebar-summary-site">Sitewide: ' +
        site.done +
        ' / ' +
        site.total +
        ' (' +
        site.percent +
        '%)</div>';
    }
    html += '</div>';

    var modules = topic.modules || [];
    if (modules.length) {
      modules.forEach(function (module, moduleIndex) {
        var chapters = topic.chapters.filter(function (ch) {
          return ch.module === module.id;
        });
        if (!chapters.length) return;
        var moduleId = 'module-' + topicId + '-' + moduleIndex;
        html += '<div class="module">';
        html +=
          '<button class="module-toggle" type="button" data-list-id="' +
          moduleId +
          '" aria-expanded="true">';
        html += '<span>' + escapeHtml(module.title) + '</span>';
        html += '<span class="chapter-caret">▶</span></button>';
        html += '<div class="module-body expanded" id="' + moduleId + '">';
        chapters.forEach(function (chapter, chapterIndex) {
          html += renderChapter(topicId, chapter, topicId + '-m' + moduleIndex + '-c' + chapterIndex);
        });
        html += '</div></div>';
      });
    } else {
      topic.chapters.forEach(function (chapter, chapterIndex) {
        html += renderChapter(topicId, chapter, topicId + '-c' + chapterIndex);
      });
    }

    sidebar.innerHTML = html;
    bindToggles(sidebar);
  }

  function renderChapter(topicId, chapter, listKey) {
    var stats = Nav.chapterStats(topicId, chapter);
    var listId = 'chapter-links-' + listKey;
    var html = '<div class="chapter" data-chapter-id="' + escapeHtml(chapter.id) + '">';
    html +=
      '<button class="chapter-toggle" type="button" data-list-id="' +
      listId +
      '" aria-expanded="false">';
    html += '<span class="chapter-toggle-main">';
    html += '<span class="chapter-title">' + escapeHtml(chapter.title) + '</span>';
    html +=
      '<span class="chapter-meta">' +
      stats.done +
      '/' +
      stats.total +
      ' · ' +
      Nav.formatMinutes(chapter.minutes);
    if (stats.quiz) {
      html += ' · quiz ' + stats.quiz.best + '/' + stats.quiz.total;
    } else if (chapter.hasQuiz) {
      html += ' · quiz pending';
    }
    html += '</span>';
    html += progressBar(stats.percent);
    html += '</span>';
    html += '<span class="chapter-caret">▶</span></button>';
    html += '<ul class="chapter-links" id="' + listId + '">';

    chapter.lessons.forEach(function (lesson, lessonIndex) {
      var id = Nav.lessonId(topicId, lesson.hash);
      var done = Progress.isDone(id);
      var visited = Progress.isVisited(id);
      var code =
        (Nav.lessonCode && Nav.lessonCode(topicId, chapter, lessonIndex + 1)) ||
        String(lessonIndex + 1);
      var classes = [];
      if (done) classes.push('done');
      else if (visited) classes.push('read');
      html += '<li><a href="#' + topicId + '/' + lesson.hash + '"';
      html += ' data-lesson-id="' + escapeHtml(id) + '"';
      html += ' class="' + classes.join(' ') + '">';
      html +=
        '<span class="lesson-check" aria-hidden="true">' +
        (done ? '✓' : '') +
        '</span>';
      html +=
        '<span class="lesson-label"><span class="lesson-num">' +
        escapeHtml(code) +
        '</span> ' +
        escapeHtml(lesson.label) +
        '</span>';
      html += '<span class="lesson-mins">' + lesson.minutes + 'm</span>';
      html += '</a></li>';
    });

    if (chapter.hasQuiz) {
      var quiz = Progress.quizScore(chapter.quizId);
      var quizClass = quiz ? 'quiz done' : 'quiz';
      html +=
        '<li><a href="#' +
        topicId +
        '/quiz/' +
        chapter.id +
        '" class="' +
        quizClass +
        '" data-quiz-id="' +
        escapeHtml(chapter.quizId) +
        '">';
      html += '<span class="lesson-check" aria-hidden="true">' + (quiz ? '✓' : '◎') + '</span>';
      html += '<span class="lesson-label">Chapter quiz</span>';
      if (quiz) {
        html +=
          '<span class="lesson-mins">' + quiz.best + '/' + quiz.total + '</span>';
      }
      html += '</a></li>';
    }

    html += '</ul></div>';
    return html;
  }

  function bindToggles(root) {
    root.querySelectorAll('.chapter-toggle, .module-toggle').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var listId = btn.getAttribute('data-list-id');
        var list = document.getElementById(listId);
        if (!list) return;
        var isExpanded = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', String(!isExpanded));
        list.classList.toggle('expanded', !isExpanded);
      });
    });
  }

  function expandForActive(parsed) {
    var selector;
    if (parsed.kind === 'lesson') {
      selector = '.sidebar a[data-lesson-id="' + Nav.lessonId(parsed.topicId, parsed.lessonHash) + '"]';
    } else if (parsed.kind === 'quiz') {
      selector = '.sidebar a[data-quiz-id="' + parsed.topicId + '/' + parsed.chapterId + '"]';
    } else {
      return;
    }
    var active = document.querySelector(selector);
    if (!active) return;

    var chapter = active.closest('.chapter');
    if (chapter) {
      var btn = chapter.querySelector('.chapter-toggle');
      var list = chapter.querySelector('.chapter-links');
      if (btn && list) {
        btn.setAttribute('aria-expanded', 'true');
        list.classList.add('expanded');
      }
    }
    var module = active.closest('.module');
    if (module) {
      var modBtn = module.querySelector('.module-toggle');
      var modBody = module.querySelector('.module-body');
      if (modBtn && modBody) {
        modBtn.setAttribute('aria-expanded', 'true');
        modBody.classList.add('expanded');
      }
    }
  }

  function paintSidebarActive(parsed) {
    document.querySelectorAll('.sidebar a').forEach(function (a) {
      a.classList.remove('active');
    });
    if (parsed.kind === 'lesson') {
      var lessonId = Nav.lessonId(parsed.topicId, parsed.lessonHash);
      var lessonLink = document.querySelector('.sidebar a[data-lesson-id="' + lessonId + '"]');
      if (lessonLink) lessonLink.classList.add('active');
    } else if (parsed.kind === 'quiz') {
      var quizLink = document.querySelector(
        '.sidebar a[data-quiz-id="' + parsed.topicId + '/' + parsed.chapterId + '"]'
      );
      if (quizLink) quizLink.classList.add('active');
    }
  }

  function continueTarget(topicId) {
    var last = Progress.lastVisited();
    if (last && last.indexOf(topicId + '/') === 0) {
      return '#' + last;
    }
    var topic = Nav.topic(topicId);
    if (!topic || !topic.chapters.length) return null;
    for (var c = 0; c < topic.chapters.length; c += 1) {
      var chapter = topic.chapters[c];
      for (var l = 0; l < chapter.lessons.length; l += 1) {
        var lesson = chapter.lessons[l];
        var id = Nav.lessonId(topicId, lesson.hash);
        if (!Progress.isDone(id)) return '#' + topicId + '/' + lesson.hash;
      }
    }
    var first = topic.chapters[0].lessons[0];
    return first ? '#' + topicId + '/' + first.hash : null;
  }

  function renderDashboard(topicId) {
    var welcome = document.getElementById('welcome');
    if (!welcome) return;
    var topic = Nav.topic(topicId);
    if (!topic) return;

    var stats = Nav.topicStats(topicId);
    var continueHref = continueTarget(topicId);

    var html = '<h1>' + escapeHtml(topic.welcomeTitle || topic.title) + '</h1>';
    html += '<p class="tagline">' + escapeHtml(topic.welcomeTagline || '') + '</p>';

    html += '<div class="dash-overview">';
    html += '<div class="dash-ring" data-pct="' + stats.percent + '">';
    html += '<strong>' + stats.percent + '%</strong>';
    html += '<span>complete</span></div>';
    html += '<div class="dash-stats">';
    html +=
      '<div><strong>' +
      stats.done +
      '</strong><span>of ' +
      stats.total +
      ' lessons done</span></div>';
    html +=
      '<div><strong>' +
      Nav.formatMinutes(stats.minutesLeft) +
      '</strong><span>reading left</span></div>';
    html +=
      '<div><strong>' +
      Nav.formatMinutes(stats.minutes) +
      '</strong><span>total in this topic</span></div>';
    html += '</div></div>';

    if (continueHref) {
      html +=
        '<p class="dash-continue"><a class="dash-cta" href="' +
        continueHref +
        '">Continue where you left off →</a></p>';
    }

    html +=
      '<p class="dash-hint">Open any lesson from the <strong>left menu</strong> — chapter progress stays there so this page stays a clean home for the topic.</p>';

    html += '<div class="dash-tools">';
    html += '<button type="button" class="dash-btn" id="dash-export">Export progress</button>';
    html += '<button type="button" class="dash-btn" id="dash-import">Import progress</button>';
    html +=
      '<button type="button" class="dash-btn dash-btn-danger" id="dash-reset">Reset my progress</button>';
    html += '<input type="file" id="dash-import-file" accept="application/json,.json" hidden>';
    html += '</div>';

    welcome.innerHTML = html;
    welcome.style.display = 'block';

    document.getElementById('dash-export').addEventListener('click', function () {
      var blob = new Blob([Progress.exportJson()], { type: 'application/json' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'backbenchlearner-progress.json';
      a.click();
      URL.revokeObjectURL(url);
    });

    document.getElementById('dash-import').addEventListener('click', function () {
      document.getElementById('dash-import-file').click();
    });

    document.getElementById('dash-import-file').addEventListener('change', function (event) {
      var file = event.target.files && event.target.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function () {
        try {
          Progress.importJson(String(reader.result));
          syncUI();
        } catch (err) {
          window.alert('Could not import that file. Make sure it is a BackbenchLearner progress export.');
        }
      };
      reader.readAsText(file);
      event.target.value = '';
    });

    document.getElementById('dash-reset').addEventListener('click', function () {
      if (
        window.confirm(
          'Reset all progress for every topic on this browser? This cannot be undone (export first if you want a backup).'
        )
      ) {
        Progress.reset();
        syncUI();
      }
    });
  }

  function renderPracticeHub() {
    var welcome = document.getElementById('welcome');
    if (!welcome) return;
    var html = '<h1>Practice</h1>';
    html +=
      '<p class="tagline">Timed module mock exams with conceptual, use-case, and numerical questions. Change answers until you submit — then review explanations and section scores.</p>';
    Nav.topics().forEach(function (topic) {
      var mocks = listModuleMocks(topic.id);
      html += '<h2 class="practice-topic-heading">' + escapeHtml(topic.navLabel) + '</h2>';
      html += '<div class="dash-chapters">';
      if (!mocks.length) {
        var qCount = topicQuestionCount(topic.id);
        html += '<div class="dash-chapter">';
        html += '<div class="dash-chapter-head"><strong>Full-topic mock</strong>';
        html += '<span>' + (qCount ? qCount + ' chapter Qs' : 'Coming soon') + '</span></div>';
        html +=
          '<p class="dash-continue" style="margin:0.75rem 0 0"><a class="dash-cta" href="' +
          (qCount ? '#practice/' + topic.id : '#' + topic.id) +
          '">' +
          (qCount ? 'Start mixed mock →' : 'Study this topic →') +
          '</a></p></div>';
      } else {
        mocks.forEach(function (mock) {
          var score = Progress.quizScore(mock.id);
          var shortTitle = mock.title
            .replace(topic.navLabel, '')
            .replace(/^[\s·\-–—]+/, '')
            .trim();
          if (!shortTitle) shortTitle = mock.title;
          html += '<div class="dash-chapter">';
          html += '<div class="dash-chapter-head">';
          html += '<strong>' + escapeHtml(shortTitle) + '</strong>';
          html +=
            '<span>' + mock.count + ' Q · ' + mock.minutes + ' min</span></div>';
          if (score) {
            html +=
              '<div class="dash-quiz-score">Best: ' +
              score.best +
              '/' +
              score.total +
              '</div>';
          } else {
            html += '<div class="dash-quiz-score muted">Not attempted yet</div>';
          }
          html +=
            '<p class="dash-continue" style="margin:0.75rem 0 0"><a class="dash-cta" href="#practice/' +
            topic.id +
            '/' +
            mock.key +
            '">Start mock exam →</a></p></div>';
        });
      }
      html += '</div>';
    });
    welcome.innerHTML = html;
    welcome.style.display = 'block';
  }

  function renderCheatHub() {
    var welcome = document.getElementById('welcome');
    if (!welcome) return;
    var html = '<h1>Cheat sheets</h1>';
    html +=
      '<p class="tagline">Topic and chapter revision guides — dense bullets you can re-read in about 30 minutes before a quiz or interview.</p>';
    Nav.topics().forEach(function (topic) {
      var sheets = listCheatSheets(topic.id);
      if (!sheets.length) return;
      html += '<h2 class="practice-topic-heading">' + escapeHtml(topic.navLabel) + '</h2>';
      html += '<div class="dash-chapters">';
      sheets.forEach(function (sheet) {
        html += '<div class="dash-chapter">';
        html += '<div class="dash-chapter-head">';
        html += '<strong>' + escapeHtml(sheet.title) + '</strong>';
        html += '<span>~' + sheet.minutes + ' min</span></div>';
        if (sheet.description) {
          html +=
            '<div class="dash-quiz-score muted">' +
            escapeHtml(sheet.description) +
            '</div>';
        }
        html +=
          '<p class="dash-continue" style="margin:0.75rem 0 0"><a class="dash-cta" href="#cheatsheet/' +
          topic.id +
          '/' +
          sheet.slug +
          '">Open revision guide →</a></p></div>';
      });
      html += '</div>';
    });
    if (html.indexOf('dash-chapter') === -1) {
      html += '<p class="tagline">No cheat sheets yet for these topics.</p>';
    }
    welcome.innerHTML = html;
    welcome.style.display = 'block';
  }

  function navMode(kind) {
    if (kind === 'practice-hub' || kind === 'practice-mock') return 'practice';
    if (kind === 'cheatsheet-hub' || kind === 'cheatsheet') return 'cheatsheet';
    return 'learn';
  }

  function syncUI() {
    var parsed = parseHash();
    var mode = navMode(parsed.kind);
    if (mode === 'learn') setCurrentTopicId(parsed.topicId);
    setHeaderActive(parsed.topicId, mode);
    renderSidebar(parsed.topicId);

    var welcome = document.getElementById('welcome');
    var frame = document.getElementById('content-frame');

    if (parsed.kind === 'practice-hub') {
      frame.src = '';
      showLessonStage(false);
      renderPracticeHub();
      closeMenus();
      return;
    }

    if (parsed.kind === 'practice-mock') {
      clearShellLessonRail();
      frame.src = practiceUrl(parsed.topicId, parsed.mockKey || '');
      showLessonStage(true);
      welcome.style.display = 'none';
      setSidebarOpen(false);
      closeMenus();
      return;
    }

    if (parsed.kind === 'cheatsheet-hub') {
      frame.src = '';
      showLessonStage(false);
      renderCheatHub();
      closeMenus();
      return;
    }

    if (parsed.kind === 'cheatsheet') {
      var sheet = findCheatSheet(parsed.topicId, parsed.sheetSlug);
      if (!sheet) {
        renderCheatHub();
        frame.src = '';
        showLessonStage(false);
        closeMenus();
        return;
      }
      clearShellLessonRail();
      frame.src = cheatsheetUrl(sheet);
      showLessonStage(true);
      welcome.style.display = 'none';
      setSidebarOpen(false);
      closeMenus();
      return;
    }

    if (parsed.kind === 'lesson') {
      var topic = Nav.topic(parsed.topicId);
      var lesson = null;
      for (var i = 0; i < topic.chapters.length; i += 1) {
        for (var j = 0; j < topic.chapters[i].lessons.length; j += 1) {
          if (topic.chapters[i].lessons[j].hash === parsed.lessonHash) {
            lesson = topic.chapters[i].lessons[j];
            break;
          }
        }
        if (lesson) break;
      }
      if (!lesson) {
        renderDashboard(parsed.topicId);
        frame.src = '';
        showLessonStage(false);
        return;
      }
      frame.src = lessonUrl(parsed.topicId, lesson);
      showLessonStage(true);
      welcome.style.display = 'none';
      setSidebarOpen(false);
    } else if (parsed.kind === 'quiz') {
      clearShellLessonRail();
      frame.src = quizUrl(parsed.topicId, parsed.chapterId);
      showLessonStage(true);
      welcome.style.display = 'none';
      setSidebarOpen(false);
    } else {
      frame.src = '';
      showLessonStage(false);
      renderDashboard(parsed.topicId);
    }

    paintSidebarActive(parsed);
    expandForActive(parsed);
    closeMenus();
  }

  function renderSeoLinks() {
    var el = document.getElementById('seo-links');
    if (!el) return;
    var html = '';
    Nav.topics().forEach(function (topic) {
      topic.chapters.forEach(function (chapter) {
        chapter.lessons.forEach(function (lesson) {
          html +=
            '<a href="' +
            lessonUrl(topic.id, lesson) +
            '">' +
            escapeHtml(topic.navLabel) +
            ': ' +
            escapeHtml(lesson.label) +
            '</a>';
        });
      });
      listCheatSheets(topic.id).forEach(function (sheet) {
        html +=
          '<a href="' +
          cheatsheetUrl(sheet) +
          '">Cheat sheet: ' +
          escapeHtml(sheet.title) +
          '</a>';
      });
    });
    el.innerHTML = html;
  }

  // ---- visitor admin ----
  // countapi.xyz is dead; use the public CountAPI successor (no signup).
  var VISITOR_COUNTER_KEY = 'backbenchlearner_com_site_visitors_v1';
  var VISITOR_COUNTER_HIT =
    'https://countapi.mileshilliard.com/api/v1/hit/' + VISITOR_COUNTER_KEY;
  var VISITOR_COUNTER_GET =
    'https://countapi.mileshilliard.com/api/v1/get/' + VISITOR_COUNTER_KEY;

  function formatLocation(data) {
    if (!data) return 'Unknown';
    var parts = [];
    if (data.city) parts.push(data.city);
    if (data.region) parts.push(data.region);
    if (data.country_name) parts.push(data.country_name);
    return parts.length ? parts.join(', ') : 'Unknown';
  }

  function parseCounterValue(data) {
    if (!data || data.value == null) return null;
    var n = Number(data.value);
    return Number.isFinite(n) ? n : null;
  }

  /** Count one visit per browser tab session (not on every admin panel open). */
  function trackSiteVisit() {
    try {
      if (sessionStorage.getItem('bbl.visit-counted') === '1') return;
      sessionStorage.setItem('bbl.visit-counted', '1');
    } catch (e) { /* still attempt hit */ }
    fetch(VISITOR_COUNTER_HIT, { method: 'GET', mode: 'cors', cache: 'no-store' }).catch(
      function () { /* ignore */ }
    );
  }

  function loadVisitorInfo() {
    var countEl = document.getElementById('visitor-count');
    var locationEl = document.getElementById('visitor-location');
    if (countEl) {
      countEl.textContent = 'Loading…';
      fetch(VISITOR_COUNTER_GET, { method: 'GET', mode: 'cors', cache: 'no-store' })
        .then(function (resp) {
          return resp.json();
        })
        .then(function (data) {
          var value = parseCounterValue(data);
          countEl.textContent = value == null ? 'N/A' : value.toLocaleString();
        })
        .catch(function () {
          countEl.textContent = 'N/A';
        });
    }
    if (locationEl) {
      fetch('https://ipapi.co/json/')
        .then(function (resp) {
          return resp.json();
        })
        .then(function (data) {
          locationEl.textContent = formatLocation(data);
        })
        .catch(function () {
          locationEl.textContent = 'Unknown';
        });
    }
  }

  function checkVisitorAuth() {
    try {
      return sessionStorage.getItem('visitor-auth') === 'true';
    } catch (e) {
      return false;
    }
  }

  function setVisitorAuth(val) {
    try {
      if (val) sessionStorage.setItem('visitor-auth', 'true');
      else sessionStorage.removeItem('visitor-auth');
    } catch (e) { /* ignore */ }
  }

  function showVisitorPanel() {
    document.getElementById('visitor-panel').classList.add('visible');
  }

  function hideVisitorPanel() {
    document.getElementById('visitor-panel').classList.remove('visible');
  }

  function promptVisitorPassword() {
    var pass = window.prompt('Admin password:');
    if (pass === 'bbl-admin') {
      setVisitorAuth(true);
      showVisitorPanel();
      loadVisitorInfo();
    } else if (pass !== null) {
      window.alert('Incorrect password');
    }
  }

  // ---- boot ----
  function bindMenuTrigger(itemId, triggerId) {
    var trigger = document.getElementById(triggerId);
    if (!trigger) return;
    trigger.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      var item = document.getElementById(itemId);
      if (item && item.classList.contains('open')) closeMenus();
      else openMenu(itemId);
    });
  }

  bindMenuTrigger('nav-learn', 'learn-trigger');
  bindMenuTrigger('nav-practice', 'practice-trigger');
  bindMenuTrigger('nav-cheatsheet', 'cheatsheet-trigger');

  document.getElementById('main-nav').addEventListener('click', function (e) {
    var topicLink = e.target.closest('a[data-topic-id]');
    if (topicLink) {
      e.preventDefault();
      closeMenus();
      var topicId = topicLink.getAttribute('data-topic-id');
      if (window.location.hash === '#' + topicId) {
        setCurrentTopicId(topicId);
        syncUI();
      } else {
        window.location.hash = topicId;
      }
      return;
    }
    var practiceLink = e.target.closest('a[data-practice-topic], a[href="#practice"]');
    if (practiceLink) {
      e.preventDefault();
      closeMenus();
      var href = practiceLink.getAttribute('href') || '';
      window.location.hash = href.replace(/^#/, '');
      return;
    }
    var cheatLink = e.target.closest('a[href^="#cheatsheet"]');
    if (cheatLink) {
      e.preventDefault();
      closeMenus();
      var cheatHref = cheatLink.getAttribute('href') || '';
      window.location.hash = cheatHref.replace(/^#/, '');
    }
  });

  var navBackdrop = document.getElementById('nav-backdrop');
  if (navBackdrop) {
    navBackdrop.addEventListener('click', closeMenus);
  }
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeMenus();
  });

  var sidebarToggle = document.getElementById('sidebar-toggle');
  var sidebarBackdrop = document.getElementById('sidebar-backdrop');
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', function () {
      setSidebarOpen(!document.getElementById('sidebar').classList.contains('open'));
    });
  }
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', function () {
      setSidebarOpen(false);
    });
  }

  window.addEventListener('hashchange', syncUI);
  Progress.subscribe(function () {
    var parsed = parseHash();
    var mode = navMode(parsed.kind);
    renderSidebar(parsed.topicId);
    paintSidebarActive(parsed);
    expandForActive(parsed);
    setHeaderActive(parsed.topicId, mode);
    if (parsed.kind === 'welcome') renderDashboard(parsed.topicId);
    if (parsed.kind === 'practice-hub') renderPracticeHub();
    if (parsed.kind === 'cheatsheet-hub') renderCheatHub();
  });

  bindShellLessonRail();
  renderSeoLinks();
  syncUI();
  trackSiteVisit();

  var unlockBtn = document.getElementById('visitor-unlock-btn');
  if (unlockBtn) {
    unlockBtn.addEventListener('click', function () {
      if (checkVisitorAuth()) {
        setVisitorAuth(false);
        hideVisitorPanel();
      } else {
        promptVisitorPassword();
      }
    });
  }
  if (checkVisitorAuth()) {
    showVisitorPanel();
    loadVisitorInfo();
  }
})();
