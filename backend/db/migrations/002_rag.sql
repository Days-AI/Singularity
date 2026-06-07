-- Project Singularity - Graph-RAG schema (Phase 1).
-- Adds an evidence_chunks vector table + match_documents RPC, and aligns the
-- report_outputs embedding dimension with the local embedding model.
--
-- The default embedding model is BAAI/bge-small-en-v1.5 (384 dims). If you
-- change EMBEDDING_MODEL_ID / EMBEDDING_DIMS, update the vector(...) sizes here
-- and re-run this migration.

create extension if not exists vector;

-- ---------------------------------------------------------------------------
-- Realign report_outputs.embedding to the local model dimension (was 1536).
-- The column is unused today, so a drop/re-add is safe.
-- ---------------------------------------------------------------------------
alter table public.report_outputs drop column if exists embedding;
alter table public.report_outputs add column embedding vector(384);

-- ---------------------------------------------------------------------------
-- evidence_chunks - embedded evidence items + report sections for retrieval
-- ---------------------------------------------------------------------------
create table if not exists public.evidence_chunks (
    id          uuid primary key default gen_random_uuid(),
    session_id  text,
    content     text not null,
    metadata    jsonb not null default '{}'::jsonb,
    embedding   vector(384),
    created_at  timestamptz not null default now()
);

create index if not exists idx_evidence_chunks_session
    on public.evidence_chunks (session_id);

-- Approximate nearest-neighbor index (cosine). Tune lists for table size.
create index if not exists idx_evidence_chunks_embedding
    on public.evidence_chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Service-role only (backend uses the service key, which bypasses RLS).
alter table public.evidence_chunks enable row level security;

-- ---------------------------------------------------------------------------
-- match_documents - cosine similarity search RPC over evidence_chunks
-- ---------------------------------------------------------------------------
create or replace function public.match_documents(
    query_embedding vector(384),
    match_count int default 5,
    filter jsonb default '{}'::jsonb
)
returns table (
    id uuid,
    session_id text,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        ec.id,
        ec.session_id,
        ec.content,
        ec.metadata,
        1 - (ec.embedding <=> query_embedding) as similarity
    from public.evidence_chunks ec
    where ec.embedding is not null
      and (filter = '{}'::jsonb or ec.metadata @> filter)
    order by ec.embedding <=> query_embedding
    limit match_count;
end;
$$;
