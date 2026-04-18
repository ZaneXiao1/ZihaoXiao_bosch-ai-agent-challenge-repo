#!/usr/bin/env python3
"""
Interactive terminal chat demo for the ME Engineering Assistant.

Usage:
    python scripts/chat_cli.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing the package first.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

from me_assistant.agent.graph import create_agent, query_agent  # noqa: E402
from me_assistant.exceptions import MEAssistantError  # noqa: E402


def main() -> int:
    """Run an interactive REPL for live terminal demos."""
    load_dotenv(Path(__file__).parent.parent / ".env")

    print("Initialising ME Engineering Assistant...")
    try:
        agent = create_agent()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to initialise agent: {exc}")
        return 1

    print("Ready. Ask a question about ECU-700 / ECU-800 documentation.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Session ended.")
            return 0

        try:
            result = query_agent(agent, question)
            print(f"Assistant> {result['answer']}\n")
        except MEAssistantError as exc:
            print(f"Assistant> Error: {exc}\n")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Assistant> Unexpected error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
