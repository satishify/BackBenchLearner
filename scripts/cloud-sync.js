/**
 * Cloud progress sync via Supabase Auth + progress table.
 * Depends on: @supabase/supabase-js (CDN), scripts/supabase-config.js, scripts/progress.js
 *
 * If BBL_SUPABASE.url / anonKey are empty, this module stays idle.
 */
(function (global) {
  'use strict';

  var BBL = (global.BBL = global.BBL || {});
  var Progress = BBL.Progress;
  var cfg = global.BBL_SUPABASE || {};
  var PUSH_DELAY_MS = 900;

  var client = null;
  var user = null;
  var pushTimer = null;
  var syncing = false;
  var lastStatus = 'off';
  var listeners = [];

  function configured() {
    return Boolean(cfg.url && cfg.anonKey && global.supabase && global.supabase.createClient);
  }

  function setStatus(status, detail) {
    lastStatus = status;
    listeners.forEach(function (fn) {
      try {
        fn({ status: status, detail: detail || '', user: user });
      } catch (e) {
        /* ignore listener errors */
      }
    });
    paintChrome();
  }

  function shortEmail(email) {
    if (!email) return 'Signed in';
    if (email.length <= 22) return email;
    return email.slice(0, 10) + '…' + email.slice(-8);
  }

  function ensureModal() {
    if (document.getElementById('bbl-auth-modal')) return;
    var wrap = document.createElement('div');
    wrap.id = 'bbl-auth-modal';
    wrap.className = 'bbl-auth-modal';
    wrap.hidden = true;
    wrap.innerHTML =
      '<div class="bbl-auth-card" role="dialog" aria-modal="true" aria-labelledby="bbl-auth-title">' +
      '<button type="button" class="bbl-auth-close" id="bbl-auth-close" aria-label="Close">×</button>' +
      '<h2 id="bbl-auth-title">Sync progress across devices</h2>' +
      '<p class="bbl-auth-lead">Sign in so lesson and quiz progress syncs across phones and PCs. Local progress is merged so you do not lose completions.</p>' +
      '<button type="button" class="bbl-auth-google" id="bbl-auth-google">' +
      '<span class="bbl-auth-google-icon" aria-hidden="true">' +
      '<svg viewBox="0 0 48 48" width="18" height="18"><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/></svg>' +
      '</span>Continue with Google</button>' +
      '<p class="bbl-auth-google-hint">Uses the Google/Gmail account already signed in on this browser when available.</p>' +
      '<div class="bbl-auth-divider"><span>or use email</span></div>' +
      '<label class="bbl-auth-label" for="bbl-auth-email">Email</label>' +
      '<input id="bbl-auth-email" class="bbl-auth-input" type="email" autocomplete="email" placeholder="you@example.com" />' +
      '<button type="button" class="bbl-auth-primary" id="bbl-auth-send">Email me a login link</button>' +
      '<p class="bbl-auth-msg" id="bbl-auth-msg" aria-live="polite"></p>' +
      '<p class="bbl-auth-note">After Google sign-in you return to this site automatically. Magic-link email still works if you prefer.</p>' +
      '</div>';
    document.body.appendChild(wrap);

    wrap.addEventListener('click', function (event) {
      if (event.target === wrap) closeModal();
    });
    document.getElementById('bbl-auth-close').addEventListener('click', closeModal);
    document.getElementById('bbl-auth-google').addEventListener('click', signInWithGoogle);
    document.getElementById('bbl-auth-send').addEventListener('click', sendMagicLink);
    document.getElementById('bbl-auth-email').addEventListener('keydown', function (event) {
      if (event.key === 'Enter') sendMagicLink();
    });
  }

  function openModal() {
    if (!configured()) {
      window.alert(
        'Cloud sync is not configured yet.\n\n' +
          '1) Create a free Supabase project\n' +
          '2) Run tools/supabase-progress.sql\n' +
          '3) Put Project URL + anon key in scripts/supabase-config.js'
      );
      return;
    }
    ensureModal();
    var modal = document.getElementById('bbl-auth-modal');
    var msg = document.getElementById('bbl-auth-msg');
    msg.textContent = '';
    modal.hidden = false;
    document.getElementById('bbl-auth-google').focus();
  }

  function closeModal() {
    var modal = document.getElementById('bbl-auth-modal');
    if (modal) modal.hidden = true;
  }

  function paintChrome() {
    var host = document.getElementById('bbl-account');
    if (!host) return;

    if (!configured()) {
      host.innerHTML =
        '<button type="button" class="bbl-account-btn" id="bbl-account-setup" title="Cloud sync setup needed">Cloud sync</button>';
      document.getElementById('bbl-account-setup').addEventListener('click', openModal);
      return;
    }

    if (!user) {
      host.innerHTML =
        '<button type="button" class="bbl-account-btn" id="bbl-account-signin">Sign in to sync</button>';
      document.getElementById('bbl-account-signin').addEventListener('click', openModal);
      return;
    }

    var label =
      lastStatus === 'saving'
        ? 'Saving…'
        : lastStatus === 'syncing'
          ? 'Syncing…'
          : lastStatus === 'error'
            ? 'Sync error'
            : 'Synced';
    host.innerHTML =
      '<span class="bbl-account-user" title="' +
      escapeAttr(user.email || '') +
      '">' +
      escapeHtml(shortEmail(user.email)) +
      '</span>' +
      '<span class="bbl-account-status" data-status="' +
      escapeAttr(lastStatus) +
      '">' +
      escapeHtml(label) +
      '</span>' +
      '<button type="button" class="bbl-account-btn" id="bbl-account-sync">Sync now</button>' +
      '<button type="button" class="bbl-account-btn bbl-account-btn-quiet" id="bbl-account-signout">Sign out</button>';

    document.getElementById('bbl-account-sync').addEventListener('click', function () {
      pullAndMerge().then(function () {
        return pushNow();
      });
    });
    document.getElementById('bbl-account-signout').addEventListener('click', signOut);
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, function (char) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[char];
    });
  }

  function escapeAttr(text) {
    return escapeHtml(text).replace(/'/g, '&#39;');
  }

  function redirectTo() {
    return global.location.origin + global.location.pathname;
  }

  async function signInWithGoogle() {
    var msg = document.getElementById('bbl-auth-msg');
    if (!client) return;
    msg.textContent = 'Opening Google…';
    var result = await client.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: redirectTo(),
        // Prefer the account already signed into this browser / show chooser quickly.
        queryParams: { prompt: 'select_account' }
      }
    });
    if (result.error) {
      msg.textContent =
        result.error.message ||
        'Google sign-in failed. Enable Google under Supabase → Authentication → Providers.';
      setStatus('error', result.error.message);
    }
  }

  async function sendMagicLink() {
    var input = document.getElementById('bbl-auth-email');
    var msg = document.getElementById('bbl-auth-msg');
    var email = (input.value || '').trim();
    if (!email || email.indexOf('@') < 0) {
      msg.textContent = 'Enter a valid email address.';
      return;
    }
    msg.textContent = 'Sending login link…';
    var result = await client.auth.signInWithOtp({
      email: email,
      options: { emailRedirectTo: redirectTo() }
    });
    if (result.error) {
      msg.textContent = result.error.message || 'Could not send login link.';
      setStatus('error', result.error.message);
      return;
    }
    msg.textContent = 'Check your email for the login link, then return here.';
  }

  async function signOut() {
    if (!client) return;
    await client.auth.signOut();
    user = null;
    setStatus('signed_out');
  }

  async function pullAndMerge() {
    if (!client || !user || !Progress) return false;
    syncing = true;
    setStatus('syncing');
    try {
      var result = await client
        .from('progress')
        .select('data, updated_at')
        .eq('user_id', user.id)
        .maybeSingle();
      if (result.error) throw result.error;
      if (result.data && result.data.data) {
        Progress.mergeState(result.data.data);
      }
      setStatus('ready');
      return true;
    } catch (err) {
      setStatus('error', (err && err.message) || 'pull failed');
      return false;
    } finally {
      syncing = false;
    }
  }

  async function pushNow() {
    if (!client || !user || !Progress) return false;
    syncing = true;
    setStatus('saving');
    try {
      var payload = {
        user_id: user.id,
        data: Progress.state(),
        updated_at: new Date().toISOString()
      };
      var result = await client.from('progress').upsert(payload, { onConflict: 'user_id' });
      if (result.error) throw result.error;
      setStatus('ready');
      return true;
    } catch (err) {
      setStatus('error', (err && err.message) || 'push failed');
      return false;
    } finally {
      syncing = false;
    }
  }

  function schedulePush() {
    if (!user || syncing) return;
    global.clearTimeout(pushTimer);
    pushTimer = global.setTimeout(function () {
      pushNow();
    }, PUSH_DELAY_MS);
  }

  async function onAuth(session) {
    user = session && session.user ? session.user : null;
    if (!user) {
      setStatus(configured() ? 'signed_out' : 'off');
      return;
    }
    closeModal();
    await pullAndMerge();
    await pushNow();
  }

  function init() {
    ensureAccountSlot();
    if (!configured()) {
      setStatus('off');
      return;
    }

    client = global.supabase.createClient(cfg.url, cfg.anonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true
      }
    });

    client.auth.getSession().then(function (result) {
      onAuth(result.data && result.data.session);
    });

    client.auth.onAuthStateChange(function (_event, session) {
      onAuth(session);
    });

    if (Progress && Progress.subscribe) {
      Progress.subscribe(function () {
        schedulePush();
      });
    }
  }

  function ensureAccountSlot() {
    if (document.getElementById('bbl-account')) return;
    var header = document.querySelector('.header-inner');
    if (!header) return;
    var slot = document.createElement('div');
    slot.id = 'bbl-account';
    slot.className = 'bbl-account';
    header.appendChild(slot);
  }

  BBL.CloudSync = {
    isConfigured: configured,
    openLogin: openModal,
    syncNow: function () {
      return pullAndMerge().then(function () {
        return pushNow();
      });
    },
    subscribe: function (fn) {
      listeners.push(fn);
      return function () {
        listeners = listeners.filter(function (item) {
          return item !== fn;
        });
      };
    },
    getUser: function () {
      return user;
    },
    status: function () {
      return lastStatus;
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
