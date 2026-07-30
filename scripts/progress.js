/**
 * BackbenchLearner progress store and curriculum navigation.
 *
 * There is no backend, so a learner's progress lives in localStorage under a
 * single versioned key. Lesson pages run inside an iframe in the shell, which
 * means two documents share one store: they stay in sync through the browser's
 * `storage` event, with postMessage as a fallback for contexts (file://,
 * private browsing) where storage is unavailable or not shared.
 *
 * Exposes window.BBL.Progress and window.BBL.Nav.
 */
(function (global) {
  'use strict';

  var BBL = (global.BBL = global.BBL || {});

  var STORAGE_KEY = 'bbl.progress.v1';
  var LEGACY_READ_KEY = 'backbenchlearner-read';
  var MESSAGE_TYPE = 'bbl:progress';

  // ----------------------------------------------------------------------
  // Storage with graceful degradation
  // ----------------------------------------------------------------------

  var memoryFallback = null;

  function storage() {
    try {
      var probe = '__bbl_probe__';
      global.localStorage.setItem(probe, '1');
      global.localStorage.removeItem(probe);
      return global.localStorage;
    } catch (e) {
      return null;
    }
  }

  function emptyState() {
    return { done: {}, visited: {}, quiz: {}, last: null };
  }

  function normalise(raw) {
    var state = emptyState();
    if (!raw || typeof raw !== 'object') return state;
    ['done', 'visited', 'quiz'].forEach(function (bucket) {
      if (raw[bucket] && typeof raw[bucket] === 'object') state[bucket] = raw[bucket];
    });
    if (typeof raw.last === 'string') state.last = raw.last;
    return state;
  }

  function read() {
    var store = storage();
    if (!store) return memoryFallback || (memoryFallback = migrateLegacy(emptyState()));
    try {
      var raw = store.getItem(STORAGE_KEY);
      if (raw) return normalise(JSON.parse(raw));
    } catch (e) {
      /* corrupt payload: fall through to a fresh state */
    }
    return migrateLegacy(emptyState());
  }

  function write(state) {
    var store = storage();
    if (!store) {
      memoryFallback = state;
      return;
    }
    try {
      store.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      memoryFallback = state;
    }
  }

  /**
   * The pre-rewrite site tracked read lessons in sessionStorage keyed by file
   * path. Carry those over once so nobody loses their place in the upgrade.
   */
  function migrateLegacy(state) {
    var legacy;
    try {
      legacy = global.sessionStorage && global.sessionStorage.getItem(LEGACY_READ_KEY);
    } catch (e) {
      return state;
    }
    if (!legacy) return state;
    try {
      JSON.parse(legacy).forEach(function (oldPath) {
        var id = BBL.Nav ? BBL.Nav.idFromLegacyPath(oldPath) : null;
        if (id) state.visited[id] = Date.now();
      });
      write(state);
      global.sessionStorage.removeItem(LEGACY_READ_KEY);
    } catch (e) {
      /* nothing worth recovering */
    }
    return state;
  }

  // ----------------------------------------------------------------------
  // Change notification
  // ----------------------------------------------------------------------

  var listeners = [];

  function notify() {
    listeners.forEach(function (fn) {
      try {
        fn();
      } catch (e) {
        /* a broken listener must not stop the others */
      }
    });
  }

  function broadcast(state) {
    var payload = { type: MESSAGE_TYPE, state: state };
    try {
      if (global.parent && global.parent !== global) global.parent.postMessage(payload, '*');
    } catch (e) {
      /* cross-origin parent: storage events will have to do */
    }
    var frame = global.document && global.document.getElementById('content-frame');
    if (frame && frame.contentWindow) {
      try {
        frame.contentWindow.postMessage(payload, '*');
      } catch (e) {
        /* frame not ready yet */
      }
    }
  }

  /** Union-merge by timestamp, so merges are order-independent and idempotent. */
  function merge(state, incoming) {
    var changed = false;
    ['done', 'visited'].forEach(function (bucket) {
      var source = incoming[bucket] || {};
      Object.keys(source).forEach(function (id) {
        if (!state[bucket][id] || state[bucket][id] < source[id]) {
          state[bucket][id] = source[id];
          changed = true;
        }
      });
    });
    Object.keys(incoming.quiz || {}).forEach(function (id) {
      var mine = state.quiz[id];
      var theirs = incoming.quiz[id];
      if (!mine || theirs.best > mine.best) {
        state.quiz[id] = theirs;
        changed = true;
      }
    });
    if (incoming.last && incoming.last !== state.last) {
      state.last = incoming.last;
      changed = true;
    }
    return changed;
  }

  function update(mutator) {
    var state = read();
    mutator(state);
    write(state);
    notify();
    broadcast(state);
  }

  if (global.addEventListener) {
    global.addEventListener('storage', function (event) {
      if (event.key === STORAGE_KEY || event.key === null) notify();
    });
    global.addEventListener('message', function (event) {
      var data = event.data;
      if (!data || data.type !== MESSAGE_TYPE || !data.state) return;
      var state = read();
      if (merge(state, data.state)) {
        write(state);
        notify();
      }
    });
  }

  // ----------------------------------------------------------------------
  // Public progress API
  // ----------------------------------------------------------------------

  var Progress = {
    STORAGE_KEY: STORAGE_KEY,

    state: read,

    subscribe: function (fn) {
      listeners.push(fn);
      return function () {
        listeners = listeners.filter(function (item) {
          return item !== fn;
        });
      };
    },

    isDone: function (id) {
      return Boolean(read().done[id]);
    },

    isVisited: function (id) {
      var state = read();
      return Boolean(state.visited[id] || state.done[id]);
    },

    setDone: function (id, value) {
      update(function (state) {
        if (value) {
          state.done[id] = Date.now();
          state.visited[id] = state.visited[id] || Date.now();
        } else {
          delete state.done[id];
        }
      });
    },

    toggleDone: function (id) {
      var next = !Progress.isDone(id);
      Progress.setDone(id, next);
      return next;
    },

    markVisited: function (id) {
      if (read().visited[id]) {
        update(function (state) {
          state.last = id;
        });
        return;
      }
      update(function (state) {
        state.visited[id] = Date.now();
        state.last = id;
      });
    },

    lastVisited: function () {
      return read().last;
    },

    quizScore: function (quizId) {
      return read().quiz[quizId] || null;
    },

    setQuizScore: function (quizId, score, total) {
      update(function (state) {
        var existing = state.quiz[quizId];
        var best = existing ? Math.max(existing.best, score) : score;
        state.quiz[quizId] = { best: best, total: total, last: score, at: Date.now() };
      });
    },

    reset: function () {
      update(function (state) {
        state.done = {};
        state.visited = {};
        state.quiz = {};
        state.last = null;
      });
    },

    exportJson: function () {
      return JSON.stringify(
        { version: 1, exportedAt: new Date().toISOString(), progress: read() },
        null,
        2
      );
    },

    importJson: function (text) {
      var parsed = JSON.parse(text);
      var incoming = normalise(parsed.progress || parsed);
      var state = read();
      merge(state, incoming);
      write(state);
      notify();
      broadcast(state);
      return true;
    }
  };

  // ----------------------------------------------------------------------
  // Curriculum navigation
  // ----------------------------------------------------------------------

  function curriculum() {
    return BBL.CURRICULUM || {};
  }

  function topicIds() {
    return BBL.TOPIC_ORDER || Object.keys(curriculum());
  }

  function lessonId(topicId, lessonHash) {
    return topicId + '/' + lessonHash;
  }

  /** Leading outline code from a chapter title, e.g. "1.1 AI & ..." → "1.1". */
  function chapterCode(topicId, chapter) {
    var match = String((chapter && chapter.title) || '').match(/^(\d+(?:\.\d+)*)\b/);
    if (match) return match[1];
    var topic = curriculum()[topicId];
    if (!topic || !chapter) return '1';
    for (var i = 0; i < topic.chapters.length; i += 1) {
      if (topic.chapters[i].id === chapter.id) return String(i + 1);
    }
    return '1';
  }

  /** Hierarchical lesson number, e.g. "1.1.12" or "3.2" for unnumbered chapters. */
  function lessonCode(topicId, chapter, indexInChapter) {
    return chapterCode(topicId, chapter) + '.' + indexInChapter;
  }

  /**
   * Flatten a topic into the order a learner walks it: every lesson in every
   * chapter, with the chapter quiz appended after its last lesson.
   */
  function sequence(topicId) {
    var topic = curriculum()[topicId];
    if (!topic) return [];
    var nodes = [];
    topic.chapters.forEach(function (chapter) {
      chapter.lessons.forEach(function (lesson, index) {
        var indexInChapter = index + 1;
        nodes.push({
          type: 'lesson',
          topicId: topicId,
          chapter: chapter,
          lesson: lesson,
          indexInChapter: indexInChapter,
          code: lessonCode(topicId, chapter, indexInChapter),
          id: lessonId(topicId, lesson.hash)
        });
      });
      if (chapter.hasQuiz) {
        nodes.push({ type: 'quiz', topicId: topicId, chapter: chapter, id: chapter.quizId });
      }
    });
    return nodes;
  }

  var Nav = {
    topicIds: topicIds,

    topic: function (topicId) {
      return curriculum()[topicId] || null;
    },

    topics: function () {
      return topicIds().map(function (id) {
        return curriculum()[id];
      });
    },

    lessonId: lessonId,

    chapterCode: chapterCode,

    lessonCode: lessonCode,

    sequence: sequence,

    chapter: function (topicId, chapterId) {
      var topic = curriculum()[topicId];
      if (!topic) return null;
      for (var i = 0; i < topic.chapters.length; i += 1) {
        if (topic.chapters[i].id === chapterId) return topic.chapters[i];
      }
      return null;
    },

    module: function (topicId, moduleId) {
      var topic = curriculum()[topicId];
      if (!topic || !topic.modules) return null;
      for (var i = 0; i < topic.modules.length; i += 1) {
        if (topic.modules[i].id === moduleId) return topic.modules[i];
      }
      return null;
    },

    /** Locate a lesson and its neighbours in the walking order. */
    locate: function (topicId, lessonHash) {
      var nodes = sequence(topicId);
      for (var i = 0; i < nodes.length; i += 1) {
        var node = nodes[i];
        if (node.type === 'lesson' && node.lesson.hash === lessonHash) {
          return {
            node: node,
            chapter: node.chapter,
            lesson: node.lesson,
            indexInChapter: node.indexInChapter,
            code: node.code,
            chapterTotal: node.chapter.lessons.length,
            prev: nodes[i - 1] || null,
            next: nodes[i + 1] || null
          };
        }
      }
      return null;
    },

    /** Map a pre-rewrite file path onto a current lesson id, for migration. */
    idFromLegacyPath: function (oldPath) {
      var decoded = oldPath;
      try {
        decoded = decodeURIComponent(oldPath);
      } catch (e) {
        /* use the raw value */
      }
      var slug = decoded.replace(/\.html$/, '').split('/').pop();
      var ids = topicIds();
      for (var t = 0; t < ids.length; t += 1) {
        var topic = curriculum()[ids[t]];
        for (var c = 0; c < topic.chapters.length; c += 1) {
          var lessons = topic.chapters[c].lessons;
          for (var l = 0; l < lessons.length; l += 1) {
            if (lessons[l].slug === slug) return lessonId(ids[t], lessons[l].hash);
          }
        }
      }
      return null;
    },

    chapterStats: function (topicId, chapter) {
      var state = read();
      var done = 0;
      var minutesDone = 0;
      chapter.lessons.forEach(function (lesson) {
        if (state.done[lessonId(topicId, lesson.hash)]) {
          done += 1;
          minutesDone += lesson.minutes;
        }
      });
      var quiz = chapter.hasQuiz ? state.quiz[chapter.quizId] || null : null;
      return {
        done: done,
        total: chapter.lessons.length,
        minutes: chapter.minutes,
        minutesDone: minutesDone,
        percent: chapter.lessons.length ? Math.round((done / chapter.lessons.length) * 100) : 0,
        quiz: quiz
      };
    },

    topicStats: function (topicId) {
      var topic = curriculum()[topicId];
      if (!topic) return { done: 0, total: 0, minutes: 0, minutesLeft: 0, percent: 0 };
      var done = 0;
      var total = 0;
      var minutes = 0;
      var minutesDone = 0;
      var self = this;
      topic.chapters.forEach(function (chapter) {
        var stats = self.chapterStats(topicId, chapter);
        done += stats.done;
        total += stats.total;
        minutes += stats.minutes;
        minutesDone += stats.minutesDone;
      });
      return {
        done: done,
        total: total,
        minutes: minutes,
        minutesDone: minutesDone,
        minutesLeft: Math.max(0, minutes - minutesDone),
        percent: total ? Math.round((done / total) * 100) : 0
      };
    },

    siteStats: function () {
      var self = this;
      return topicIds().reduce(
        function (acc, id) {
          var stats = self.topicStats(id);
          acc.done += stats.done;
          acc.total += stats.total;
          acc.minutes += stats.minutes;
          acc.minutesLeft += stats.minutesLeft;
          acc.percent = acc.total ? Math.round((acc.done / acc.total) * 100) : 0;
          return acc;
        },
        { done: 0, total: 0, minutes: 0, minutesLeft: 0, percent: 0 }
      );
    },

    /** "1h 05m" for long spans, "45 min" for short ones. */
    formatMinutes: function (minutes) {
      if (minutes < 60) return minutes + ' min';
      var hours = Math.floor(minutes / 60);
      var rest = minutes % 60;
      return hours + 'h ' + (rest < 10 ? '0' : '') + rest + 'm';
    }
  };

  BBL.Progress = Progress;
  BBL.Nav = Nav;
})(window);
