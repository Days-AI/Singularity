"""Per-agent synthetic memory backstory seeded from agent index."""
from __future__ import annotations

import random

from agents.cognitive.types import AgentMemory

_BRAND_TOUCHES = [
    "positive brand encounter at retail",
    "neutral online ad exposure",
    "negative customer service experience",
    "heard mixed reviews from friends",
    "no prior brand exposure",
]
_CAMPAIGNS = [
    "saw a social media campaign last month",
    "ignored a recent email promotion",
    "participated in a loyalty program",
    "unsubscribed from marketing list",
    "no campaign memory",
]
_PURCHASES = [
    "bought similar product before",
    "abandoned cart once",
    "compared alternatives extensively",
    "impulse purchase history",
    "never purchased in category",
]
_SOCIAL = [
    "friends mostly positive online",
    "influencer mentioned the category",
    "family skeptical of new products",
    "colleagues discussing adoption",
    "no social chatter noticed",
]


def generate_memory(agent_index: int, archetype_id: str) -> AgentMemory:
    rng = random.Random(agent_index * 997 + hash(archetype_id) % 10000)
    brand = rng.choice(_BRAND_TOUCHES)
    campaigns = [rng.choice(_CAMPAIGNS)]
    purchases = [rng.choice(_PURCHASES)]
    social = [rng.choice(_SOCIAL)]

    trust_mod = 0.0
    if "negative" in brand.lower():
        trust_mod -= 0.35
    elif "positive" in brand.lower():
        trust_mod += 0.2
    if "skeptical" in social[0].lower():
        trust_mod -= 0.1

    return AgentMemory(
        past_campaigns=campaigns,
        brand_experience=[brand],
        purchases=purchases,
        social_exposure=social,
        trust_modifier=trust_mod,
    )


def update_memory_after_stimulus(memory: AgentMemory, sentiment: float, query: str) -> None:
    """Lightweight within-run memory write."""
    snippet = query[:60].strip() or "recent stimulus"
    if sentiment > 0.2:
        memory.brand_experience.append(f"positive reaction to: {snippet}")
        memory.trust_modifier = min(1.0, memory.trust_modifier + 0.05)
    elif sentiment < -0.2:
        memory.brand_experience.append(f"negative reaction to: {snippet}")
        memory.trust_modifier = max(-1.0, memory.trust_modifier - 0.08)
    memory.past_campaigns.append(f"evaluated: {snippet}")
