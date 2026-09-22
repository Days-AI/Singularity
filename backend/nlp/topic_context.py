"""Run-level topic context from query + evidence (KeyBERT, spaCy, embeddings)."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from config import get_settings
from nlp import embeddings
from state import EvidenceItem

logger = logging.getLogger("singularity.nlp.topic_context")

_nlp = None
_nlp_failed = False
_keybert = None
_keybert_failed = False


@dataclass
class TopicContext:
    query: str
    keyphrases: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    domain_label: str = "general"

    def keyphrase_str(self, limit: int = 5) -> str:
        return ", ".join(self.keyphrases[:limit])

    def evidence_hook(self) -> str:
        return self.evidence_snippets[0] if self.evidence_snippets else ""


def _load_spacy():
    global _nlp, _nlp_failed
    if _nlp_failed:
        return None
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        model = get_settings().spacy_model
        _nlp = spacy.load(model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("spaCy unavailable (%s); entity extraction skipped", exc)
        _nlp_failed = True
        _nlp = None
    return _nlp


def _load_keybert():
    global _keybert, _keybert_failed
    if _keybert_failed:
        return None
    if _keybert is not None:
        return _keybert
    if not embeddings.embeddings_available():
        _keybert_failed = True
        return None
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        model_id = get_settings().keybert_model_id
        st = SentenceTransformer(model_id, device="cpu")
        _keybert = KeyBERT(model=st)
    except Exception as exc:  # noqa: BLE001
        logger.warning("KeyBERT unavailable: %s", exc)
        _keybert_failed = True
        _keybert = None
    return _keybert


def _fallback_keyphrases(text: str, k: int = 8) -> list[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stop = {
        "about", "with", "from", "that", "this", "what", "when", "where",
        "which", "their", "would", "could", "should", "market", "analyze",
    }
    seen: list[str] = []
    for w in words:
        if w not in stop and w not in seen:
            seen.append(w)
        if len(seen) >= k:
            break
    return seen


def _extract_entities(text: str) -> list[str]:
    nlp = _load_spacy()
    if not nlp:
        return []
    doc = nlp(text[:8000])
    return list({ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1})[:12]


def _extract_keyphrases(corpus: str, k: int = 8) -> list[str]:
    kb = _load_keybert()
    if kb and corpus.strip():
        try:
            kws = kb.extract_keywords(corpus, keyphrase_ngram_range=(1, 3), top_n=k)
            return [str(kw[0]) for kw in kws if kw[0]]
        except Exception as exc:  # noqa: BLE001
            logger.debug("KeyBERT extract failed: %s", exc)
    return _fallback_keyphrases(corpus, k)


def _domain_label(keyphrases: list[str], query: str) -> str:
    text = " ".join(keyphrases + [query]).lower()
    if re.search(r"brand|reputation|trust|campaign", text):
        return "brand perception"
    if re.search(r"product|launch|feature|pricing", text):
        return "product launch"
    if re.search(r"policy|regul|government|public", text):
        return "policy & regulation"
    if re.search(r"market|consumer|audience|sentiment", text):
        return "market sentiment"
    return "consumer research"


async def _semantic_snippets(
    query: str,
    evidence: list[EvidenceItem],
    k: int = 3,
) -> list[str]:
    if not evidence or not embeddings.embeddings_available():
        return [e.title for e in evidence[:k]]
    texts = [f"{e.title}. {e.detail}"[:400] for e in evidence[:20]]
    q_vec = await embeddings.embed_query(query)
    doc_vecs = await embeddings.embed_texts(texts)
    if not q_vec or not doc_vecs:
        return [e.title for e in evidence[:k]]
    scores = []
    for i, dv in enumerate(doc_vecs):
        sim = sum(a * b for a, b in zip(q_vec, dv))
        scores.append((sim, texts[i]))
    scores.sort(key=lambda x: -x[0])
    return [s[1][:180] for s in scores[:k]]


def _build_blocking(query: str, evidence: list[EvidenceItem]) -> TopicContext:
    parts = [query]
    for e in evidence[:12]:
        parts.append(e.title)
        if e.detail:
            parts.append(e.detail[:200])
    corpus = " ".join(parts)
    keyphrases = _extract_keyphrases(corpus)
    entities = _extract_entities(corpus)
    domain = _domain_label(keyphrases, query)
    return TopicContext(
        query=query,
        keyphrases=keyphrases,
        entities=entities,
        evidence_snippets=[],
        domain_label=domain,
    )


async def build_topic_context(query: str, evidence: list[EvidenceItem]) -> TopicContext:
    """Build topic context; no-op minimal context when NLP pipeline disabled."""
    settings = get_settings()
    if not settings.nlp_pipeline_enabled:
        return TopicContext(query=query, domain_label="general")

    ctx = await asyncio.to_thread(_build_blocking, query, evidence)
    ctx.evidence_snippets = await _semantic_snippets(query, evidence)
    return ctx
