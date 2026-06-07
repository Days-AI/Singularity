"""Centralized configuration for the Singularity backend.

All runtime settings are sourced from environment variables (see .env.example).
Everything has a sane default so the backend boots even with no .env: missing
external services degrade gracefully rather than crashing the flow.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Ollama / local Gemma -------------------------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434")
    # Verified installed tag on this machine.
    ollama_model: str = Field(default="gemma4:latest")
    ollama_timeout_s: float = Field(default=120.0)
    ollama_num_ctx: int = Field(default=4096)
    # Max in-flight Ollama requests. CPU-only Gemma: 1–2. GPU Ollama: 8–16 typical.
    # Hard ceiling for ALL Gemma calls (DAG, archetype, cognitive NLG, council).
    ollama_concurrency: int = Field(default=2)

    # --- Agentic LLM throughput / latency tuning -----------------------------
    # Output-token caps keep structured calls from over-decoding (the single
    # biggest agentic-latency lever). Each is a ceiling, not a target; the model
    # stops early once it emits a complete object.
    dag_max_tokens: int = Field(default=640)         # DAG decomposition JSON
    persona_max_tokens: int = Field(default=256)     # per-archetype persona JSON
    report_polish_max_tokens: int = Field(default=1024)  # executive report JSON

    # --- OpenRouter polishing layer ------------------------------------------
    openrouter_api_key: str | None = Field(default=None)
    openrouter_model: str = Field(default="google/gemma-2-27b-it")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    # When False (default), report polish stays on local Ollama Gemma even if
    # OPENROUTER_API_KEY is set. Set USE_OPENROUTER_POLISH=true to enable cloud.
    use_openrouter_polish: bool = Field(default=False)

    # --- Supabase persistence (optional) -------------------------------------
    supabase_url: str | None = Field(default=None)
    # Accept either SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY.
    supabase_service_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("supabase_service_key", "supabase_service_role_key"),
    )
    supabase_anon_key: str | None = Field(default=None)

    # --- Company intelligence (Gemma-orchestrated yfinance) -------------------
    # The financial evidence agent resolves the query to specific public companies
    # via local Gemma, then yfinance supplies qualitative, non-price company data
    # (profile, ESG, governance, leadership, ownership, analyst sentiment, news).
    company_intel_max_companies: int = Field(default=3)
    company_intel_max_items_per_company: int = Field(default=4)

    # --- External data APIs ---------------------------------------------------
    serper_api_key: str | None = Field(default=None)
    # Parallel web API (https://parallel.ai) - search/extraction tool.
    parallel_api_key: str | None = Field(default=None)
    parallel_base_url: str = Field(default="https://api.parallel.ai")
    # Default tier for the ChatParallelWeb web-grounded chat model ("speed", etc.).
    parallel_chat_model: str = Field(default="speed")

    # --- GDELT global news/event search (optional, flag-gated) ----------------
    # When enabled (and gdeltdoc is installed), the GDELT fetcher contributes
    # recent news articles + an article-volume time series to evidence routing.
    # Keyless. Default OFF so existing behavior is unchanged until turned on.
    gdelt_enabled: bool = Field(default=False)
    # Rolling lookback window (days) used to build the GDELT date filter.
    gdelt_lookback_days: int = Field(default=10)
    # Maximum number of GDELT articles surfaced as evidence items per query.
    gdelt_max_articles: int = Field(default=2)

    # --- Native LangChain search wrappers (optional, flag-gated) --------------
    # When enabled (and the packages are installed), the Serper / DuckDuckGo /
    # Parallel evidence fetchers route through the official LangChain tool classes
    # in tools.search_providers instead of the hand-rolled httpx/ddgs path. Falls
    # back transparently to the legacy fetchers when disabled or unavailable.
    langchain_native_tools: bool = Field(default=False)

    # --- LangChain tool layer + Graph-RAG (all optional, flag-gated) ----------
    # When enabled, evidence collection routes through the LangChain tool layer
    # (yfinance, arxiv, wikipedia, serper, parallel, duckduckgo). Falls back to
    # the hand-rolled evidence path when disabled or when langchain is missing.
    langchain_enabled: bool = Field(default=False)
    # When enabled, evidence + report sections are embedded into Supabase
    # pgvector and a GraphRAG retriever augments synthesis with prior context.
    rag_enabled: bool = Field(default=False)
    # Local HuggingFace sentence-transformers embedding model (CPU-friendly).
    embedding_model_id: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dims: int = Field(default=384)
    rag_top_k: int = Field(default=5)

    # --- CrewAI synthesis layer (optional, flag-gated) -----------------------
    # When enabled, the report stage is produced by a CrewAI crew injected with
    # aggregated persona OCEAN/facet context. Falls back to the two-stage
    # Gemma->polish report.build() when disabled or crewai is missing.
    crewai_enabled: bool = Field(default=False)
    crew_max_personas_per_cluster: int = Field(default=1)

    # --- Hugging Face (Days234 personality engine, gated) --------------------
    hf_token: str | None = Field(default=None)
    personality_repo_id: str = Field(default="Days234/personality-engine")
    # If true, skip the HF download/inference entirely and use the analytic
    # OCEAN fallback (useful offline / when the gated repo is inaccessible).
    disable_personality_engine: bool = Field(default=False)

    # --- Report synthesis -----------------------------------------------------
    # Max evidence items passed to report LLM and External Intelligence section.
    report_evidence_max_items: int = Field(default=12)

    # --- App / flow tuning ----------------------------------------------------
    app_secret_key: str = Field(default="dev-secret-change-me-min-32-characters")
    allowed_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000")
    auth_enabled: bool = Field(default=False)

    persona_population: int = Field(default=1500)
    persona_batches: int = Field(default=6)
    # K-means cluster centroids for IPIP expansion — NOT per-agent LLM count.
    # With COGNITIVE_AGENTS_ENABLED + IPIP, archetype LLM is skipped; keep <= 64.
    persona_archetypes: int = Field(default=36)
    # Real IPIP-300 baseline (CSV with OCEAN + 30 facet columns). If present, the
    # psychometric engine clusters it into archetypes and uses the real profiles
    # as the population; otherwise it synthesizes a lattice.
    ipip_data_path: str | None = Field(default=None)
    max_concurrent_agents: int = Field(default=8)
    forecast_horizon_days: int = Field(default=90)

    # --- Entropy-driven cognitive agents --------------------------------------
    cognitive_agents_enabled: bool = Field(default=True)
    cognitive_llm_sample_size: int = Field(default=150)
    # Effective parallelism is min(this, ollama_concurrency).
    cognitive_llm_concurrency: int = Field(default=8)
    cognitive_deliberation_max_tokens: int = Field(default=384)
    cognitive_run_seed: int | None = Field(default=None)

    # --- Social simulation + specialist council + consensus -------------------
    social_simulation_enabled: bool = Field(default=True)
    social_simulation_rounds: int = Field(default=3)
    specialist_council_enabled: bool = Field(default=True)
    council_polish_openrouter: bool = Field(default=True)
    consensus_engine_enabled: bool = Field(default=True)

    # --- TimesFM / forecast ---------------------------------------------------
    timesfm_model_id: str = Field(default="google/timesfm-2.5-200m-pytorch")
    timesfm_max_context: int = Field(default=1024)

    # --- Core decision intelligence engine ------------------------------------
    monte_carlo_simulations: int = Field(default=2000)
    swarm_iterations: int = Field(default=50)
    # When True, decision stages may call local Gemma for narrative enrichment.
    decision_engine_llm_enrich: bool = Field(default=True)

    # --- Master log (global JSONL audit trail) --------------------------------
    master_log_enabled: bool = Field(default=True)
    master_log_path: str = Field(default="logs/master.jsonl")
    master_log_heartbeat_s: int = Field(default=60)

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def parallel_enabled(self) -> bool:
        return bool(self.parallel_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
