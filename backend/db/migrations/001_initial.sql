-- Project Singularity - initial schema (spec section 5).
-- Sessions, report outputs, forecast results + pgvector + RLS.
--
-- NOTE: This was applied to the live project (igbmyhdnoosjbsutaykd) via the
-- Supabase integration. The spec's `agent_profiles` table already existed in
-- that project with an incompatible schema, and the backend does not persist
-- per-persona rows, so it is intentionally NOT (re)created here to avoid
-- clobbering existing data. Re-add it under a distinct name if per-persona
-- persistence is needed later.

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- simulation_sessions
-- ---------------------------------------------------------------------------
create table if not exists public.simulation_sessions (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid references auth.users (id) on delete cascade,
    query       text not null,
    dag_json    jsonb,
    status      text not null default 'pending',  -- pending|running|complete|failed
    flow_uuid   text unique,
    created_at  timestamptz not null default now()
);

create index if not exists idx_sim_sessions_user on public.simulation_sessions (user_id);
create index if not exists idx_sim_sessions_flow on public.simulation_sessions (flow_uuid);

-- ---------------------------------------------------------------------------
-- report_outputs - McKinsey-grade sections + chart payloads + embeddings
-- ---------------------------------------------------------------------------
create table if not exists public.report_outputs (
    id            uuid primary key default gen_random_uuid(),
    session_id    uuid references public.simulation_sessions (id) on delete cascade,
    section       text,
    content_md    text,
    chart_data    jsonb,
    causal_graph  jsonb,
    forecast_data jsonb,
    embedding     vector(1536),              -- semantic search over past reports
    created_at    timestamptz not null default now()
);

create index if not exists idx_sim_reports_session on public.report_outputs (session_id);

-- ---------------------------------------------------------------------------
-- forecast_results
-- ---------------------------------------------------------------------------
create table if not exists public.forecast_results (
    id           uuid primary key default gen_random_uuid(),
    session_id   uuid references public.simulation_sessions (id) on delete cascade,
    model_used   text,
    horizon      int,
    predictions  jsonb,
    mase_score   double precision,
    created_at   timestamptz not null default now()
);

create index if not exists idx_sim_forecasts_session on public.forecast_results (session_id);

-- ---------------------------------------------------------------------------
-- Row Level Security (scoped to the authenticated role)
-- ---------------------------------------------------------------------------
alter table public.simulation_sessions enable row level security;
alter table public.report_outputs      enable row level security;
alter table public.forecast_results    enable row level security;

drop policy if exists "users own sim sessions" on public.simulation_sessions;
create policy "users own sim sessions" on public.simulation_sessions
    for all to authenticated using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "users own sim reports" on public.report_outputs;
create policy "users own sim reports" on public.report_outputs
    for all to authenticated using (
        exists (select 1 from public.simulation_sessions s
                where s.id = report_outputs.session_id and s.user_id = auth.uid())
    );

drop policy if exists "users own sim forecasts" on public.forecast_results;
create policy "users own sim forecasts" on public.forecast_results
    for all to authenticated using (
        exists (select 1 from public.simulation_sessions s
                where s.id = forecast_results.session_id and s.user_id = auth.uid())
    );
