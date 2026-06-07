"""CrewAI agent + task builders and LLM wiring.

Generation agents run on local Ollama/Gemma (via litellm's ``ollama/`` provider);
the Report Editor uses OpenRouter when configured for the polishing pass, else
falls back to the same local model. All CrewAI imports happen lazily in
``crew.crew`` so importing this module never hard-requires crewai.
"""
from __future__ import annotations

import logging

from config import get_settings

logger = logging.getLogger("singularity.crew.agents")


def build_generation_llm():
    """Local Ollama/Gemma LLM for the generation agents."""
    from crewai import LLM

    settings = get_settings()
    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        temperature=0.5,
    )


def build_polish_llm():
    """OpenRouter LLM for the editorial polish pass when enabled, else local."""
    from crewai import LLM

    settings = get_settings()
    if settings.openrouter_enabled and settings.use_openrouter_polish:
        return LLM(
            model=f"openrouter/{settings.openrouter_model}",
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            temperature=0.5,
        )
    return build_generation_llm()


def build_agents(generation_llm, polish_llm, tools: list | None = None):
    """Construct the four role-specialized agents.

    Tools are attached to the Evidence Analyst when compatible; if CrewAI rejects
    them, the analyst is rebuilt without tools so the crew still runs.
    """
    from crewai import Agent

    def _agent(role, goal, backstory, llm, agent_tools=None):
        kwargs = dict(role=role, goal=goal, backstory=backstory, llm=llm,
                      verbose=False, allow_delegation=False)
        if agent_tools:
            try:
                return Agent(tools=agent_tools, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent tool attach failed (%s); building without tools", exc)
        return Agent(**kwargs)

    analyst = _agent(
        role="Evidence Analyst",
        goal="Extract the most decision-relevant facts from the evidence and retrieved context.",
        backstory="A meticulous research analyst who grounds every claim in the provided data.",
        llm=generation_llm,
        agent_tools=tools,
    )
    psychologist = _agent(
        role="Behavioral Psychologist",
        goal="Interpret the population's OCEAN and facet-level signals into actionable behavioral insight.",
        backstory="A quantitative psychologist specialized in Big Five personality and consumer behavior.",
        llm=generation_llm,
    )
    strategist = _agent(
        role="Strategy Synthesizer",
        goal="Translate evidence and behavioral insight into strategic implications and risks.",
        backstory="A communications and consumer-insights strategist who translates behavioral data into clear, audience-focused recommendations.",
        llm=generation_llm,
    )
    editor = _agent(
        role="Report Editor",
        goal="Produce the final executive report as strict JSON, preserving all quantitative findings.",
        backstory="A senior brand communications director who writes crisp, confident, audience-focused strategic prose.",
        llm=polish_llm,
    )
    return {"analyst": analyst, "psychologist": psychologist,
            "strategist": strategist, "editor": editor}


def build_tasks(agents: dict, *, evidence_brief: str, persona_brief: str,
                rag_context: str, query: str):
    """Build the sequential task chain ending in a strict-JSON report."""
    from crewai import Task

    rag_block = f"\n\nRetrieved prior context:\n{rag_context}" if rag_context else ""

    t_analyst = Task(
        description=(
            f"Analyze the evidence for the query: '{query}'.\n\n"
            f"Evidence and quantitative data:\n{evidence_brief}{rag_block}\n\n"
            "List the 4-6 most decision-relevant findings in McKinsey bullet format:\n"
            "**Insight** — supporting evidence [Source: Title] — strategic implication.\n"
            "Cite concrete numbers. Do not invent figures or URLs."
        ),
        expected_output=(
            "McKinsey-style bulleted findings with [Source: Title] citations and numbers."
        ),
        agent=agents["analyst"],
    )
    t_psych = Task(
        description=(
            "Interpret the simulated persona population's behavioral signal.\n\n"
            f"Aggregated persona context:\n{persona_brief}\n\n"
            "Explain how each cluster's OCEAN profile and dominant facet drivers shape "
            "their stance, in 3-5 sentences. Reference cluster sizes and sentiment."
        ),
        expected_output="A short behavioral interpretation grounded in OCEAN/facet aggregates.",
        agent=agents["psychologist"],
    )
    t_strategy = Task(
        description=(
            "Using the evidence findings and behavioral interpretation, derive strategic "
            "implications and the key risks. Be specific and directional."
        ),
        expected_output="Strategic implications plus a list of risk flags.",
        agent=agents["strategist"],
        context=[t_analyst, t_psych],
    )
    t_editor = Task(
        description=(
            "Compile the final executive report. Respond ONLY with valid JSON matching:\n"
            "{\n"
            '  "executive_summary": "<=150 words, pyramid: answer first>",\n'
            '  "external_intelligence_summary": "<=120 words, cites 2-3 sources>",\n'
            '  "key_findings": ["**Insight** — evidence (Source: X) — implication", "... 5-7"],\n'
            '  "strategic_implications": "<=200 words",\n'
            '  "risk_flags": ["...", "[VERIFY] when inconsistent"],\n'
            '  "application_playbooks": [\n'
            '    {"domain": "Communications & PR|Product|Branding|Social Media|Marketing|Public Policy",\n'
            '     "recommendation": "<simulation-grounded 1-2 sentences>"}\n'
            "  ]\n"
            "}\n"
            "application_playbooks MUST have exactly six entries (one per domain). "
            "Preserve all quantitative findings. Flag inconsistencies with [VERIFY]. "
            "Use McKinsey voice: short sentences, active voice, no hedging."
        ),
        expected_output=(
            "A single JSON object with executive_summary, external_intelligence_summary, "
            "key_findings, strategic_implications, risk_flags, and application_playbooks."
        ),
        agent=agents["editor"],
        context=[t_analyst, t_psych, t_strategy],
    )
    return [t_analyst, t_psych, t_strategy, t_editor]
