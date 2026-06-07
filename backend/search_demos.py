"""Interactive CLI demo for the LangChain search providers.

Run it from the ``backend/`` directory:

    python search_demos.py                # interactive menu
    python search_demos.py --choice 1 --query "What is LangChain?"
    python search_demos.py --choice 5 --urls https://example.com https://parallel.ai

Menu:

    1. DuckDuckGo Search
    2. DuckDuckGo Results
    3. Google Serper Search
    4. Parallel Search
    5. Parallel Extract
    6. Parallel Chat

The demo loads environment variables from the nearest ``.env`` (backend/.env then
repo-root .env), configures logging, and routes each menu choice to the matching
reusable function in :mod:`tools.search_providers`. Provider failures (missing API
keys or packages) are caught and reported instead of crashing the CLI.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Callable

try:  # Standalone convenience: load .env so the demo works without the FastAPI app.
    from dotenv import load_dotenv

    load_dotenv()  # backend/.env (cwd)
    load_dotenv("../.env")  # repo-root .env, if present
except Exception:  # noqa: BLE001 - dotenv is optional at runtime
    pass

from tools.search_providers import (
    ProviderError,
    duckduckgo_results,
    duckduckgo_search,
    parallel_chat,
    parallel_extract,
    parallel_search,
    serper_search,
)

logger = logging.getLogger("singularity.search_demos")

MENU = """
==================== Search Providers Demo ====================
1. DuckDuckGo Search
2. DuckDuckGo Results
3. Google Serper Search
4. Parallel Search
5. Parallel Extract
6. Parallel Chat
0. Exit
===============================================================
"""

DEFAULT_QUERY = "What is LangChain?"
DEFAULT_URLS = ["https://en.wikipedia.org/wiki/Artificial_intelligence"]


def _print_result(label: str, result: Any) -> None:
    """Pretty-print a provider result, JSON-encoding structured payloads."""
    print(f"\n----- {label} -----")
    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(result)
    print("-" * (len(label) + 12))


def _run_choice(choice: str, *, query: str, urls: list[str]) -> bool:
    """Dispatch a single menu choice. Returns ``False`` when the user chose to exit."""
    try:
        if choice == "1":
            _print_result("DuckDuckGo Search", duckduckgo_search(query))
        elif choice == "2":
            _print_result("DuckDuckGo Results", duckduckgo_results(query))
        elif choice == "3":
            _print_result("Google Serper Search", serper_search(query))
        elif choice == "4":
            _print_result("Parallel Search", parallel_search(query))
        elif choice == "5":
            _print_result("Parallel Extract", parallel_extract(urls))
        elif choice == "6":
            _print_result("Parallel Chat", parallel_chat(query))
        elif choice in {"0", "q", "quit", "exit"}:
            return False
        else:
            print(f"Unknown choice: {choice!r}. Pick 1-6 (or 0 to exit).")
    except ProviderError as exc:
        print(f"\n[provider unavailable] {exc}")
    except Exception as exc:  # noqa: BLE001 - keep the CLI alive on unexpected errors
        logger.exception("Unexpected error while running choice %s", choice)
        print(f"\n[unexpected error] {exc}")
    return True


def _interactive_loop() -> None:
    """Run the interactive menu until the user exits."""
    print(MENU)
    while True:
        try:
            choice = input("Select an option (0-6): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if choice in {"0", "q", "quit", "exit"}:
            print("Goodbye.")
            return

        if choice in {"5"}:
            raw = input("Enter URLs (space-separated) [default sample]: ").strip()
            urls = raw.split() if raw else DEFAULT_URLS
            _run_choice(choice, query=DEFAULT_QUERY, urls=urls)
        elif choice in {"1", "2", "3", "4", "6"}:
            q = input(f"Enter query [default: {DEFAULT_QUERY!r}]: ").strip() or DEFAULT_QUERY
            _run_choice(choice, query=q, urls=DEFAULT_URLS)
        else:
            print(f"Unknown choice: {choice!r}. Pick 1-6 (or 0 to exit).")
        print(MENU)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo the LangChain search providers (DuckDuckGo, Serper, Parallel).",
    )
    parser.add_argument(
        "--choice",
        choices=[str(i) for i in range(1, 7)],
        help="Run a single provider non-interactively (1-6) instead of the menu.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query/objective/question for choices 1-4 and 6.",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        default=DEFAULT_URLS,
        help="One or more URLs for choice 5 (Parallel Extract).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...). Default: INFO.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.choice:
        _run_choice(args.choice, query=args.query, urls=args.urls)
        return 0

    _interactive_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
