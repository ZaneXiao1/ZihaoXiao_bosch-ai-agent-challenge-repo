#!/usr/bin/env python3
"""
Run the RAG answer-quality evaluation and produce a report.

Usage:
    python scripts/run_eval.py                                # run extended (60Q)
    python scripts/run_eval.py --dataset official            # run official (10Q)
    python scripts/run_eval.py --dataset extended            # run extended (60Q)
    python scripts/run_eval.py --csv path/to.csv            # custom test set
    python scripts/run_eval.py --output results.json        # save raw results

The script will:
1. Initialise the RAG agent
2. Run each test question, measure latency
3. Use an LLM judge to score each answer on 5 dimensions
4. Print a summary table and save detailed results to JSON
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the src directory is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from me_assistant.agent.graph import create_agent
from me_assistant.evaluation.evaluator import (
    DEFAULT_TEST_SET,
    DIMENSIONS,
    run_evaluation,
    summarize_results,
)

_TESTS_DIR = Path(__file__).parent.parent / "tests"
_DATASET_PATHS = {
    "official": DEFAULT_TEST_SET,
    "extended": _TESTS_DIR / "test_queries_extended.csv",
}


def print_report(results, summary):
    """Print a human-readable evaluation report to stdout."""
    max_score = len(DIMENSIONS) * 5

    print("\n" + "=" * 90)
    print("  RAG Answer Quality Evaluation Report")
    print("=" * 90)

    # Per-question detail table
    print(f"\n{'ID':>3} | {'Category':<30} | {'Score':>6} | {'Lat(s)':>7} | "
          f"{'Cor':>3} {'Com':>3} {'Fai':>3} {'Rel':>3} {'Fmt':>3} | Status")
    print("-" * 105)

    for r in results:
        cor = r.correctness.score if r.correctness else "-"
        com = r.completeness.score if r.completeness else "-"
        fai = r.faithfulness.score if r.faithfulness else "-"
        rel = r.relevance.score if r.relevance else "-"
        fmt = r.format_compliance.score if r.format_compliance else "-"
        status = "OK" if not r.judge_error else f"ERR: {r.judge_error[:30]}"
        print(f"{r.question_id:>3} | {r.category:<30} | "
              f"{r.total_score:>2}/{r.max_possible:<3} | {r.latency_seconds:>6.2f}s | "
              f"{cor:>3} {com:>3} {fai:>3} {rel:>3} {fmt:>3} | {status}")

    # Summary
    print("\n" + "-" * 90)
    print("  SUMMARY")
    print("-" * 90)
    da = summary["dimension_averages"]
    print(f"  Total questions:     {summary['total_questions']}")
    print(f"  Overall avg score:   {summary['overall_avg_score']}/{summary['overall_max_score']}")
    print(f"  Correctness avg:     {da['correctness']}/5")
    print(f"  Completeness avg:    {da['completeness']}/5")
    print(f"  Faithfulness avg:    {da['faithfulness']}/5")
    print(f"  Relevance avg:       {da['relevance']}/5")
    print(f"  Format Compliance:   {da['format_compliance']}/5")
    print(f"  Latency avg/min/max: {summary['latency_avg']}s / "
          f"{summary['latency_min']}s / {summary['latency_max']}s")

    # Per-category
    print(f"\n  Per-Category Breakdown:")
    for cat, stats in summary["category_breakdown"].items():
        print(f"    {cat:<35} avg={stats['avg_score']:>5.1f}/{max_score}  "
              f"latency={stats['avg_latency']:.2f}s  (n={stats['count']})")

    print("=" * 90 + "\n")


def _resolve_csv_path(dataset: str, csv_path: str | None) -> Path:
    """Return the CSV path to evaluate.

    Custom ``--csv`` takes precedence. Otherwise the named built-in dataset
    is used.
    """
    if csv_path:
        return Path(csv_path)
    return _DATASET_PATHS[dataset]


def main():
    parser = argparse.ArgumentParser(description="RAG Answer Quality Evaluation")
    parser.add_argument(
        "--dataset",
        choices=sorted(_DATASET_PATHS),
        default="extended",
        help="Built-in dataset to run when --csv is not provided (default: extended).",
    )
    parser.add_argument("--csv", default=None, help="Path to test CSV file")
    parser.add_argument("--output", default="eval_results.json",
                        help="Output JSON file for detailed results (default: eval_results.json)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    csv_path = _resolve_csv_path(args.dataset, args.csv)

    print("Initialising agent...")
    agent = create_agent()

    print(f"Running evaluation on: {csv_path}")
    results = run_evaluation(agent, csv_path=str(csv_path))
    summary = summarize_results(results)

    # Print human-readable report
    print_report(results, summary)

    # Save detailed results to JSON
    output_path = Path(args.output)
    output_data = {
        "dataset": args.dataset if not args.csv else "custom",
        "csv_path": str(csv_path),
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed results saved to: {output_path.resolve()}")

    # Also print per-question rationales for review
    print("\n--- Judge Rationales ---")
    for r in results:
        print(f"\nQ{r.question_id}: {r.question}")
        print(f"  Model answer (truncated): {r.model_answer[:150]}...")
        for dim in DIMENSIONS:
            d = getattr(r, dim)
            if d:
                print(f"  {dim}: {d.score}/5 — {d.rationale}")


if __name__ == "__main__":
    main()
