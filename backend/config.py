"""Centralized configuration for the Singularity backend.

All runtime settings are sourced from environment variables (see .env.example).
Everything has a sane default so the backend boots even with no .env: missing
external services degrade gracefully rather than crashing the flow.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LatencyMode = Literal["fast", "balanced", "quality"]

_LATENCY_PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "cognitive_llm_sample_size": 96,
        "persona_polish_openrouter": False,
        "cognitive_deliberation_max_tokens": 160,
        "decision_engine_llm_enrich": False,
        "council_polish_openrouter": False,
        "monte_carlo_simulations": 800,
        "use_openrouter_polish": False,
    },
    "balanced": {
        "cognitive_llm_sample_size": 150,
        "persona_polish_openrouter": False,
        "cognitive_deliberation_max_tokens": 192,
        "decision_engine_llm_enrich": False,
        "council_polish_openrouter": False,
        "monte_carlo_simulations": 1000,
        "use_openrouter_polish": False,
        "flow_budget_seconds": 600,
        "programmatic_nlg_max_tokens": 72,
        "naturalize_pack_size": 6,
        "naturalize_ollama_num_ctx": 2048,
        "naturalize_pack_max_tokens": 320,
    },
}


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
    # Context window for cognitive persona NLG (falls back to ollama_num_ctx when unset).
    cognitive_ollama_num_ctx: int | None = Field(default=None)
    # Max chars for cognitive LLM user prompt before budget trimmer runs.
    cognitive_prompt_max_chars: int = Field(default=900)
    # Max in-flight Ollama requests. CPU-only Gemma: 1–2. GPU Ollama: 8–16 typical.
    # Hard ceiling for ALL Gemma calls (DAG, archetype, cognitive NLG, council).
    ollama_concurrency: int = Field(default=8)

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
    # When Ollama is down, route generate/generate_json through OpenRouter free models.
    # Independent of USE_OPENROUTER_POLISH. Requires OPENROUTER_API_KEY.
    ollama_openrouter_fallback: bool = Field(default=True)
    openrouter_fallback_models: str = Field(
        default=(
            "nvidia/nemotron-3-ultra-550b-a55b:free,"
            "google/gemma-4-31b-it:free,"
            "google/gemma-4-26b-a4b-it:free,"
            "openai/gpt-oss-20b:free"
        )
    )
    # Cap in-flight OpenRouter fallback calls (free tier is ~20 RPM).
    openrouter_fallback_concurrency: int = Field(default=3)
    ollama_circuit_failures: int = Field(default=2)
    ollama_circuit_cooldown_s: float = Field(default=30.0)

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
    # fast | balanced | quality — balanced targets sub-10-minute end-to-end runs.
    latency_mode: str = Field(
        default="balanced",
        validation_alias=AliasChoices("latency_mode", "singularity_latency_mode"),
    )
    flow_budget_seconds: int = Field(default=600)

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
    cognitive_llm_sample_size: int = Field(default=750)
    # --- NLP natural-language pipeline ----------------------------------------
    nlp_pipeline_enabled: bool = Field(default=True)
    spacy_model: str = Field(default="en_core_web_sm")
    keybert_model_id: str = Field(default="all-MiniLM-L6-v2")
    emotion_model_id: str = Field(default="j-hartmann/emotion-english-distilroberta-base")
    persona_top_p_min: float = Field(default=0.85)
    persona_top_p_max: float = Field(default=0.98)
    persona_polish_openrouter: bool = Field(default=True)
    persona_polish_concurrency: int = Field(default=8)
    bertopic_min_comments: int = Field(default=200)
    vader_sentiment_max_delta: float = Field(default=0.4)
    # Effective parallelism is min(this, ollama_concurrency).
    cognitive_llm_concurrency: int = Field(default=8)
    cognitive_deliberation_max_tokens: int = Field(default=256)
    cognitive_run_seed: int | None = Field(default=None)
    # gemma_naturalize | draft_only — programmatic path uses local Gemma by default.
    programmatic_nlg_mode: str = Field(default="gemma_naturalize")
    programmatic_nlg_max_tokens: int = Field(default=96)
    # Packed naturalize: briefs per Ollama call (4 ≈ 3× fewer HTTP round-trips).
    naturalize_pack_size: int = Field(default=1)
    # Smaller ctx for short naturalize JSON (faster decode, more VRAM headroom).
    naturalize_ollama_num_ctx: int | None = Field(default=None)
    naturalize_pack_max_tokens: int = Field(default=384)

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

    @model_validator(mode="after")
    def _apply_latency_presets(self) -> Self:
        mode = self.latency_mode_normalized
        if mode != "quality":
            preset = _LATENCY_PRESETS.get(mode, _LATENCY_PRESETS["balanced"])
            for key, value in preset.items():
                object.__setattr__(self, key, value)
        if self.cognitive_llm_concurrency > self.ollama_concurrency:
            object.__setattr__(self, "cognitive_llm_concurrency", self.ollama_concurrency)
        return self

    @property
    def latency_mode_normalized(self) -> str:
        raw = (self.latency_mode or "balanced").lower().strip()
        if raw in _LATENCY_PRESETS or raw == "quality":
            return raw
        return "balanced"

    @property
    def skip_cognitive_nlg_extras(self) -> bool:
        """Skip sentiment retry, HF emotion, and persona polish on LLM samples."""
        return self.latency_mode_normalized != "quality"

    def flow_budget_exceeded(self, elapsed_ms: int) -> bool:
        return elapsed_ms >= self.flow_budget_seconds * 1000

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
    def openrouter_fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.openrouter_fallback_models.split(",") if m.strip()]

    @property
    def ollama_fallback_enabled(self) -> bool:
        return bool(self.ollama_openrouter_fallback and self.openrouter_api_key)

    @property
    def parallel_enabled(self) -> bool:
        return bool(self.parallel_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    """Clear cached settings (tests / hot reload)."""
    get_settings.cache_clear()
