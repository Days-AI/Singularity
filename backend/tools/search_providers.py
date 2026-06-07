"""Reusable, provider-agnostic search functions built on official LangChain tools.

This module is the single home for the six first-class search providers requested
by the product spec, each wrapping an *official* LangChain integration class:

    | Function              | LangChain class                                  |
    | --------------------- | ------------------------------------------------ |
    | ``duckduckgo_search`` | ``DuckDuckGoSearchRun``                          |
    | ``duckduckgo_results``| ``DuckDuckGoSearchResults``                      |
    | ``serper_search``     | ``GoogleSerperResults`` + ``GoogleSerperAPIWrapper`` |
    | ``parallel_search``   | ``ParallelWebSearchTool``                        |
    | ``parallel_extract``  | ``ParallelExtractTool``                          |
    | ``parallel_chat``     | ``ChatParallelWeb``                              |

Design goals (consistent with the rest of the backend):

* **Import-guarded** - the optional LangChain packages (``langchain-community``,
  ``langchain-parallel``) may be absent; the module still imports and every
  function degrades to a logged, typed ``ProviderError`` rather than crashing.
* **Flag/key aware** - Serper and Parallel require API keys sourced from
  :func:`config.get_settings`; missing keys raise a clear ``ProviderError``.
* **Sync + async** - each provider exposes a blocking function and an ``a``-prefixed
  awaitable variant (the async variant offloads blocking tool calls to a worker
  thread via :func:`asyncio.to_thread` to keep the event loop free).
* **Tool reuse** - tool instances are cached with :func:`functools.lru_cache` so we
  do not re-instantiate (and re-validate) heavy LangChain objects per call.
* **Evidence bridge** - :func:`to_evidence_items` converts raw provider output into
  the project's :class:`state.EvidenceItem` shape for the flag-gated wire-in inside
  :mod:`tools.sources`.

Nothing here is imported at backend boot unless ``LANGCHAIN_NATIVE_TOOLS`` is on or
the CLI demo (``search_demos.py``) is run, so the default runtime is unaffected.
"""
from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any

from config import get_settings
from state import EvidenceItem

logger = logging.getLogger("singularity.tools.search_providers")


# --- Optional dependency probing --------------------------------------------
# ``langchain-community`` ships the DuckDuckGo + Serper wrappers; ``langchain-parallel``
# ships the Parallel search/extract tools and the ChatParallelWeb chat model. Both
# are optional: probe at import time and surface a clear error only when a missing
# provider is actually invoked.
try:  # DuckDuckGo + Serper live in langchain-community.
    from langchain_community.tools import (  # type: ignore
        DuckDuckGoSearchResults,
        DuckDuckGoSearchRun,
    )
    from langchain_community.tools.google_serper import GoogleSerperResults  # type: ignore
    from langchain_community.utilities import GoogleSerperAPIWrapper  # type: ignore

    LC_COMMUNITY_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - optional dependency
    DuckDuckGoSearchResults = None  # type: ignore
    DuckDuckGoSearchRun = None  # type: ignore
    GoogleSerperResults = None  # type: ignore
    GoogleSerperAPIWrapper = None  # type: ignore
    LC_COMMUNITY_AVAILABLE = False
    logger.debug("langchain-community not available: %s", exc)

try:  # Parallel tools + chat model live in langchain-parallel.
    from langchain_parallel import ParallelExtractTool, ParallelWebSearchTool  # type: ignore
    from langchain_parallel.chat_models import ChatParallelWeb  # type: ignore

    LC_PARALLEL_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - optional dependency
    ParallelExtractTool = None  # type: ignore
    ParallelWebSearchTool = None  # type: ignore
    ChatParallelWeb = None  # type: ignore
    LC_PARALLEL_AVAILABLE = False
    logger.debug("langchain-parallel not available: %s", exc)

try:  # HumanMessage is part of langchain-core (a transitive dep of the above).
    from langchain_core.messages import HumanMessage  # type: ignore

    LC_CORE_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - optional dependency
    HumanMessage = None  # type: ignore
    LC_CORE_AVAILABLE = False
    logger.debug("langchain-core not available: %s", exc)


# Default result cap, kept aligned with the existing fetchers in ``tools.sources``.
_DEFAULT_MAX_RESULTS = 6
_MAX_DETAIL_CHARS = 240
_MAX_TITLE_CHARS = 140


class ProviderError(RuntimeError):
    """Raised when a search provider is unavailable or its call fails.

    Carries a human-readable message describing *why* the provider could not run
    (missing package, missing API key, or an upstream API failure) so callers can
    log/surface it without re-classifying the failure.
    """


# --- Tool factories (cached) -------------------------------------------------
@lru_cache(maxsize=1)
def _ddg_run_tool() -> Any:
    """Return a cached :class:`DuckDuckGoSearchRun` instance."""
    if not LC_COMMUNITY_AVAILABLE:
        raise ProviderError(
            "DuckDuckGoSearchRun requires 'langchain-community' (pip install langchain-community ddgs)."
        )
    return DuckDuckGoSearchRun()


@lru_cache(maxsize=2)
def _ddg_results_tool(output_format: str = "list") -> Any:
    """Return a cached :class:`DuckDuckGoSearchResults` instance.

    ``output_format`` controls how the tool serializes results: ``"list"`` (default)
    yields native Python objects, ``"json"`` a JSON string, and ``"string"`` the
    legacy comma-separated form.
    """
    if not LC_COMMUNITY_AVAILABLE:
        raise ProviderError(
            "DuckDuckGoSearchResults requires 'langchain-community' (pip install langchain-community ddgs)."
        )
    return DuckDuckGoSearchResults(output_format=output_format)


@lru_cache(maxsize=1)
def _serper_tool() -> Any:
    """Return a cached :class:`GoogleSerperResults` instance bound to the API key."""
    if not LC_COMMUNITY_AVAILABLE:
        raise ProviderError(
            "GoogleSerperResults requires 'langchain-community' (pip install langchain-community)."
        )
    api_key = get_settings().serper_api_key
    if not api_key:
        raise ProviderError("SERPER_API_KEY is not configured; cannot run Google Serper search.")
    wrapper = GoogleSerperAPIWrapper(serper_api_key=api_key)
    return GoogleSerperResults(api_wrapper=wrapper)


@lru_cache(maxsize=1)
def _parallel_search_tool() -> Any:
    """Return a cached :class:`ParallelWebSearchTool` instance."""
    if not LC_PARALLEL_AVAILABLE:
        raise ProviderError(
            "ParallelWebSearchTool requires 'langchain-parallel' (pip install langchain-parallel)."
        )
    if not get_settings().parallel_api_key:
        raise ProviderError("PARALLEL_API_KEY is not configured; cannot run Parallel search.")
    # The tool reads PARALLEL_API_KEY from the environment by default; pass it
    # explicitly so it works regardless of how settings were loaded.
    return ParallelWebSearchTool(api_key=get_settings().parallel_api_key)


@lru_cache(maxsize=1)
def _parallel_extract_tool() -> Any:
    """Return a cached :class:`ParallelExtractTool` instance."""
    if not LC_PARALLEL_AVAILABLE:
        raise ProviderError(
            "ParallelExtractTool requires 'langchain-parallel' (pip install langchain-parallel)."
        )
    if not get_settings().parallel_api_key:
        raise ProviderError("PARALLEL_API_KEY is not configured; cannot run Parallel extract.")
    return ParallelExtractTool(api_key=get_settings().parallel_api_key)


@lru_cache(maxsize=4)
def _parallel_chat_model(model: str) -> Any:
    """Return a cached :class:`ChatParallelWeb` instance for ``model``."""
    if not LC_PARALLEL_AVAILABLE:
        raise ProviderError(
            "ChatParallelWeb requires 'langchain-parallel' (pip install langchain-parallel)."
        )
    if not LC_CORE_AVAILABLE:
        raise ProviderError("ChatParallelWeb requires 'langchain-core' for message types.")
    if not get_settings().parallel_api_key:
        raise ProviderError("PARALLEL_API_KEY is not configured; cannot run Parallel chat.")
    return ChatParallelWeb(model=model, api_key=get_settings().parallel_api_key)


# --- Provider 1/2: DuckDuckGo -----------------------------------------------
def duckduckgo_search(query: str) -> str:
    """Run a DuckDuckGo web search and return a single concatenated text blob.

    Wraps :class:`langchain_community.tools.DuckDuckGoSearchRun`. Keyless.

    Args:
        query: Natural-language search query.

    Returns:
        Plain-text search summary as produced by the tool.

    Raises:
        ProviderError: If ``langchain-community`` / ``ddgs`` are missing or the
            underlying search call fails.
    """
    if not query or not query.strip():
        raise ProviderError("duckduckgo_search requires a non-empty query.")
    try:
        result = _ddg_run_tool().invoke(query)
        logger.info("duckduckgo_search ok (query=%r, chars=%d)", query, len(str(result)))
        return str(result)
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("duckduckgo_search failed: %s", exc)
        raise ProviderError(f"DuckDuckGo search failed: {exc}") from exc


def duckduckgo_results(query: str, output_format: str = "list") -> list[dict[str, Any]] | str:
    """Run a DuckDuckGo search returning structured results (title/link/snippet).

    Wraps :class:`langchain_community.tools.DuckDuckGoSearchResults`. Keyless.

    Args:
        query: Natural-language search query.
        output_format: ``"list"`` (default) for native objects, ``"json"`` for a
            JSON string, or ``"string"`` for the legacy comma-separated form.

    Returns:
        A list of result dicts when ``output_format="list"``, otherwise a string.

    Raises:
        ProviderError: If the package is missing or the search call fails.
    """
    if not query or not query.strip():
        raise ProviderError("duckduckgo_results requires a non-empty query.")
    try:
        result = _ddg_results_tool(output_format).invoke(query)
        count = len(result) if isinstance(result, list) else 1
        logger.info("duckduckgo_results ok (query=%r, items=%d)", query, count)
        return result
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("duckduckgo_results failed: %s", exc)
        raise ProviderError(f"DuckDuckGo results failed: {exc}") from exc


# --- Provider 3: Google Serper ----------------------------------------------
def serper_search(query: str) -> dict[str, Any]:
    """Run a Google search via Serper and return the raw structured payload.

    Wraps :class:`langchain_community.tools.google_serper.GoogleSerperResults`
    backed by :class:`GoogleSerperAPIWrapper`. Requires ``SERPER_API_KEY``.

    Args:
        query: Natural-language search query.

    Returns:
        The decoded Serper response (dict with ``organic``/``knowledgeGraph``/etc.).
        Some tool versions return a JSON string; it is decoded to a dict for the
        caller's convenience, falling back to ``{"raw": <str>}`` if undecodable.

    Raises:
        ProviderError: If the package/key are missing or the search call fails.
    """
    if not query or not query.strip():
        raise ProviderError("serper_search requires a non-empty query.")
    try:
        result = _serper_tool().run(query)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (ValueError, TypeError):
                result = {"raw": result}
        logger.info("serper_search ok (query=%r)", query)
        return result if isinstance(result, dict) else {"raw": result}
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("serper_search failed: %s", exc)
        raise ProviderError(f"Google Serper search failed: {exc}") from exc


# --- Provider 4/5: Parallel search + extract --------------------------------
def parallel_search(objective: str, *, max_results: int = 10) -> dict[str, Any]:
    """Run a Parallel web search for a research objective.

    Wraps :class:`langchain_parallel.ParallelWebSearchTool`. Requires
    ``PARALLEL_API_KEY``.

    Args:
        objective: Natural-language description of the research goal.
        max_results: Maximum number of results to request (1-40).

    Returns:
        The tool's structured search response.

    Raises:
        ProviderError: If the package/key are missing or the search call fails.
    """
    if not objective or not objective.strip():
        raise ProviderError("parallel_search requires a non-empty objective.")
    try:
        payload: dict[str, Any] = {"objective": objective, "max_results": max_results}
        result = _parallel_search_tool().invoke(payload)
        logger.info("parallel_search ok (objective=%r)", objective)
        return result if isinstance(result, dict) else {"results": result}
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("parallel_search failed: %s", exc)
        raise ProviderError(f"Parallel search failed: {exc}") from exc


def parallel_extract(urls: list[str]) -> dict[str, Any]:
    """Extract clean, structured content from one or more web pages via Parallel.

    Wraps :class:`langchain_parallel.ParallelExtractTool`. Requires
    ``PARALLEL_API_KEY``.

    Args:
        urls: List of absolute URLs to extract content from.

    Returns:
        The tool's structured extraction response.

    Raises:
        ProviderError: If the package/key are missing, ``urls`` is empty, or the
            extraction call fails.
    """
    if not urls:
        raise ProviderError("parallel_extract requires at least one URL.")
    try:
        result = _parallel_extract_tool().invoke({"urls": urls})
        logger.info("parallel_extract ok (urls=%d)", len(urls))
        return result if isinstance(result, dict) else {"results": result}
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("parallel_extract failed: %s", exc)
        raise ProviderError(f"Parallel extract failed: {exc}") from exc


# --- Provider 6: Parallel chat (web-grounded) -------------------------------
def parallel_chat(question: str, *, model: str | None = None) -> str:
    """Ask a question to Parallel's web-grounded chat model and return the answer.

    Wraps :class:`langchain_parallel.chat_models.ChatParallelWeb`. Note that this
    chat model does *not* support tool-calling; it is a research assistant invoked
    directly with a single :class:`HumanMessage`. Requires ``PARALLEL_API_KEY``.

    Args:
        question: The user's question.
        model: Parallel chat model tier (defaults to ``settings.parallel_chat_model``,
            typically ``"speed"``).

    Returns:
        The assistant's response text.

    Raises:
        ProviderError: If the package/key are missing or the chat call fails.
    """
    if not question or not question.strip():
        raise ProviderError("parallel_chat requires a non-empty question.")
    chosen = model or get_settings().parallel_chat_model
    try:
        chat = _parallel_chat_model(chosen)
        response = chat.invoke([HumanMessage(content=question)])
        content = getattr(response, "content", response)
        logger.info("parallel_chat ok (model=%s)", chosen)
        return str(content)
    except ProviderError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("parallel_chat failed: %s", exc)
        raise ProviderError(f"Parallel chat failed: {exc}") from exc


# --- Async variants ----------------------------------------------------------
# Each async variant offloads the blocking provider call to a worker thread. This
# keeps the FastAPI event loop responsive when these are used from the wire-in.
async def aduckduckgo_search(query: str) -> str:
    """Async variant of :func:`duckduckgo_search`."""
    return await asyncio.to_thread(duckduckgo_search, query)


async def aduckduckgo_results(
    query: str, output_format: str = "list"
) -> list[dict[str, Any]] | str:
    """Async variant of :func:`duckduckgo_results`."""
    return await asyncio.to_thread(duckduckgo_results, query, output_format)


async def aserper_search(query: str) -> dict[str, Any]:
    """Async variant of :func:`serper_search`."""
    return await asyncio.to_thread(serper_search, query)


async def aparallel_search(objective: str, *, max_results: int = 10) -> dict[str, Any]:
    """Async variant of :func:`parallel_search`."""
    return await asyncio.to_thread(lambda: parallel_search(objective, max_results=max_results))


async def aparallel_extract(urls: list[str]) -> dict[str, Any]:
    """Async variant of :func:`parallel_extract`."""
    return await asyncio.to_thread(parallel_extract, urls)


async def aparallel_chat(question: str, *, model: str | None = None) -> str:
    """Async variant of :func:`parallel_chat`."""
    return await asyncio.to_thread(lambda: parallel_chat(question, model=model))


# --- Evidence bridge ---------------------------------------------------------
def _clip(text: Any, limit: int) -> str:
    return str(text or "")[:limit]


def _coerce_result_list(raw: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of a list of result dicts from heterogeneous payloads.

    Handles the shapes returned by the different providers: a bare list, a dict with
    ``results``/``organic`` keys, or a JSON string.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        for key in ("results", "organic", "items", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def to_evidence_items(
    raw: Any, *, source: str, max_items: int = _DEFAULT_MAX_RESULTS
) -> list[EvidenceItem]:
    """Convert heterogeneous provider output into :class:`state.EvidenceItem` objects.

    Maps common field aliases across providers:

    * title  <- ``title``
    * detail <- ``snippet`` | ``body`` | ``excerpts`` | ``content`` | ``description``
    * url    <- ``url`` | ``link`` | ``href``

    Args:
        raw: Provider output (list, dict, or JSON string).
        source: Label stamped onto each :class:`EvidenceItem` (e.g. ``"Serper"``).
        max_items: Maximum number of items to return.

    Returns:
        A list of normalized :class:`EvidenceItem` objects (possibly empty).
    """
    results = _coerce_result_list(raw)
    items: list[EvidenceItem] = []
    for r in results[:max_items]:
        detail = (
            r.get("snippet")
            or r.get("body")
            or r.get("excerpts")
            or r.get("content")
            or r.get("description")
            or ""
        )
        if isinstance(detail, list):
            detail = " ".join(str(d) for d in detail)
        url = r.get("url") or r.get("link") or r.get("href")
        title = r.get("title") or url or ""
        items.append(
            EvidenceItem(
                source=source,
                title=_clip(title, _MAX_TITLE_CHARS),
                detail=_clip(detail, _MAX_DETAIL_CHARS),
                url=url,
            )
        )
    return items


__all__ = [
    "ProviderError",
    "LC_COMMUNITY_AVAILABLE",
    "LC_PARALLEL_AVAILABLE",
    "LC_CORE_AVAILABLE",
    "duckduckgo_search",
    "duckduckgo_results",
    "serper_search",
    "parallel_search",
    "parallel_extract",
    "parallel_chat",
    "aduckduckgo_search",
    "aduckduckgo_results",
    "aserper_search",
    "aparallel_search",
    "aparallel_extract",
    "aparallel_chat",
    "to_evidence_items",
]
