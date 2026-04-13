"""
Evaluation script for the ME Engineering Assistant.

Runs all 10 predefined test queries, prints each question + answer,
and reports overall accuracy (pass ≥ 8/10) and per-query timing.

Usage:
    python scripts/run_evaluation.py
"""
import csv
import logging
import os
import sys
import time
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env before any other import so config.py picks up the variables
from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent.parent / ".env")

from me_assistant.agent.graph import create_agent, query_agent  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _preflight_check() -> None:
    """Verify required environment variables are set before making any API calls.

    Exits with a clear error message if configuration is missing so the user
    does not wait through agent initialisation only to see an auth error.
    """
    provider = os.getenv("MODEL_PROVIDER", "openai")
    errors: list[str] = []

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            errors.append(
                "  OPENAI_API_KEY is not set.\n"
                "  -> Get a key at https://platform.openai.com/api-keys\n"
                "  -> Then add it to your .env file:  OPENAI_API_KEY=sk-..."
            )
    elif provider == "databricks":
        for var in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "LLM_MODEL_NAME", "EMBEDDING_MODEL_NAME"):
            if not os.getenv(var):
                errors.append(f"  {var} is not set (required for MODEL_PROVIDER=databricks)")
    else:
        errors.append(f"  MODEL_PROVIDER='{provider}' is not recognised. Use 'openai' or 'databricks'.")

    if errors:
        print("=" * 72)
        print("  PRE-FLIGHT CHECK FAILED — fix the following before running:")
        print("=" * 72)
        for msg in errors:
            print(msg)
        print()
        print("  Copy .env.example to .env and fill in the missing values:")
        print("    cp .env.example .env")
        print()
        sys.exit(1)

QUERIES_PATH = Path(__file__).parent.parent / "tests" / "test_queries.csv"

# Minimum set of substrings that must appear in the answer for a question to pass.
# Checks are case-insensitive.
_PASS_KEYWORDS: dict[str, list[str]] = {
    "1": ["85"],
    "2": ["2 gb", "2gb", "lpddr4"],
    "3": ["npu", "5 tops"],
    "4": ["npu", "4 gb", "4gb", "1.5 ghz"],
    "5": ["single", "dual", "1 mbps", "2 mbps"],
    "6": ["1.7a", "1.7 a", "550ma", "550 ma"],
    "7": ["not supported", "ecu-750", "ecu-850"],
    "8": ["2 mb", "16 gb", "32 gb"],
    "9": ["105", "+105"],
    "10": ["me-driver-ctl", "--enable-npu", "--mode=performance"],
}


def _passes(question_id: str, answer: str) -> bool:
    """Return True if the answer contains at least one required keyword."""
    keywords = _PASS_KEYWORDS.get(question_id, [])
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def _load_queries() -> list[dict]:
    """Load test queries from the CSV file."""
    with open(QUERIES_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_evaluation() -> None:
    """Execute all 10 test queries and print a formatted summary report."""
    _preflight_check()

    print("=" * 72)
    print("  ME Engineering Assistant — Tier 1 Evaluation Suite")
    print("=" * 72)
    print()

    print("Initialising agent (vector stores + model loading)...")
    t_init = time.monotonic()
    agent = create_agent()
    print(f"Agent ready in {time.monotonic() - t_init:.1f}s\n")

    queries = _load_queries()
    results: list[dict] = []

    for row in queries:
        qid = row["Question_ID"]
        category = row["Category"]
        question = row["Question"]
        expected = row["Expected_Answer"]

        print(f"Q{qid}  [{category}]")
        print(f"  Question : {question}")

        t0 = time.monotonic()
        answer = query_agent(agent, question)
        elapsed = time.monotonic() - t0

        passed = _passes(qid, answer)
        status = "PASS" if passed else "FAIL"
        slow_flag = "  *** SLOW ***" if elapsed >= 10.0 else ""

        preview = answer[:300] + ("…" if len(answer) > 300 else "")
        print(f"  Answer   : {preview}")
        print(f"  Expected : {expected[:120]}")
        print(f"  Result   : [{status}]  {elapsed:.2f}s{slow_flag}")
        print()

        results.append({"id": qid, "passed": passed, "time": elapsed})

    passed_count = sum(1 for r in results if r["passed"])
    under_10_count = sum(1 for r in results if r["time"] < 10.0)
    total = len(results)
    tier1_verdict = "PASS" if passed_count >= 8 else "FAIL"
    failed_ids = [r["id"] for r in results if not r["passed"]]

    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Accuracy  : {passed_count}/{total} correct")
    print(f"  Timing    : {under_10_count}/{total} under 10 s")
    print(f"  Tier 1    : {tier1_verdict}  (threshold: 8/{total})")
    if failed_ids:
        print(f"  Failed IDs: {', '.join(failed_ids)}")
    print()


if __name__ == "__main__":
    run_evaluation()
