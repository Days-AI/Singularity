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

    # --- OpenRouter polishing layer ------------------------------------------
    openrouter_api_key: str | None = Field(default=None)
    openrouter_model: str = Field(default="google/gemma-2-27b-it")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    # --- Supabase persistence (optional) -------------------------------------
    supabase_url: str | None = Field(default=None)
    # Accept either SUPABASE_SERVICE_KEY or SUPABASE_SERVICE_ROLE_KEY.
    supabase_service_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("supabase_service_key", "supabase_service_role_key"),
    )
    supabase_anon_key: str | None = Field(default=None)

    # --- External data APIs ---------------------------------------------------
    serper_api_key: str | None = Field(default=None)

    # --- Hugging Face (Days234 personality engine, gated) --------------------
    hf_token: str | None = Field(default=None)
    personality_repo_id: str = Field(default="Days234/personality-engine")
    # If true, skip the HF download/inference entirely and use the analytic
    # OCEAN fallback (useful offline / when the gated repo is inaccessible).
    disable_personality_engine: bool = Field(default=False)

    # --- App / flow tuning ----------------------------------------------------
    app_secret_key: str = Field(default="dev-secret-change-me-min-32-characters")
    allowed_origins: str = Field(default="http://localhost:3000,http://127.0.0.1:3000")
    auth_enabled: bool = Field(default=False)

    persona_population: int = Field(default=1500)
    persona_batches: int = Field(default=6)
    # Number of representative archetypes actually simulated with the LLM.
    persona_archetypes: int = Field(default=36)
    # Real IPIP-300 baseline (CSV with OCEAN + 30 facet columns). If present, the
    # psychometric engine clusters it into archetypes and uses the real profiles
    # as the population; otherwise it synthesizes a lattice.
    ipip_data_path: str | None = Field(default=None)
    max_concurrent_agents: int = Field(default=8)
    forecast_horizon_days: int = Field(default=90)

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
