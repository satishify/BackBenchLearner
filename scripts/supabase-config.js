/**
 * Supabase cloud sync config for BackbenchLearner.
 *
 * 1. Create a free project at https://supabase.com
 * 2. Run the SQL in tools/supabase-progress.sql
 * 3. Auth → Providers → enable Email (magic link) and Google
 *    For Google: create OAuth client in Google Cloud Console, then paste
 *    Client ID + Client Secret into Supabase Google provider settings.
 *    Authorized redirect URI must be:
 *      https://YOUR_PROJECT.supabase.co/auth/v1/callback
 * 4. Auth → URL configuration → add your site URL
 *    (https://backbenchlearner.com and http://localhost:... if needed)
 * 5. Project Settings → API → paste Project URL and anon public key below
 *
 * The anon key is meant to be public in the browser. Row Level Security
 * (in the SQL file) is what keeps each user to their own progress row.
 *
 * Leave both empty to keep the site fully local (no login UI sync).
 */
window.BBL_SUPABASE = {
  url: 'https://qbodaepuntuogoygtnas.supabase.co',
  // Project Settings → API → "anon" "public" key (safe for browser with RLS)
  anonKey: 'sb_publishable_NVU1nQyYbo3SvxpOlU1PGw_1nOZMHam'
};
