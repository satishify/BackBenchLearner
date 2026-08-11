-- BackbenchLearner progress sync (Supabase)
-- Run this once in: Supabase Dashboard → SQL Editor → New query

create table if not exists public.progress (
  user_id uuid primary key references auth.users (id) on delete cascade,
  data jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.progress enable row level security;

drop policy if exists "users can read own progress" on public.progress;
create policy "users can read own progress"
  on public.progress for select
  using (auth.uid() = user_id);

drop policy if exists "users can insert own progress" on public.progress;
create policy "users can insert own progress"
  on public.progress for insert
  with check (auth.uid() = user_id);

drop policy if exists "users can update own progress" on public.progress;
create policy "users can update own progress"
  on public.progress for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Optional helper: keep updated_at fresh
create or replace function public.set_progress_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists progress_set_updated_at on public.progress;
create trigger progress_set_updated_at
  before update on public.progress
  for each row execute function public.set_progress_updated_at();
