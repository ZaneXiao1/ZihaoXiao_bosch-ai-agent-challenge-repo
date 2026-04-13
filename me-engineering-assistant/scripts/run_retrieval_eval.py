#!/usr/bin/env python3
"""
Run the RAG retrieval quality evaluation.

Usage:
    python scripts/run_retrieval_eval.py                          # default k=3
    python scripts/run_retrieval_eval.py --k 5                    # override k
    python scripts/run_retrieval_eval.py --output retrieval.json  # save results

For each test question, queries the vector stores directly and shows:
- Which chunks were retrieved
- Their rank and similarity score
- Source file and section title
- A content preview

This is for manual inspection — you review the output to check whether
the most relevant chunks are ranked at the top.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from me_assistant.evaluation.retrieval_eval import (
    run_retrieval_evaluation,
    print_retrieval_report,
)


def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Quality Evaluation")
    parser.add_argument("--csv", default=None, help="Path to test CSV file")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks per store (default: 3)")
    parser.add_argument("--output", default="retrieval_eval_results.json",
                        help="Output JSON file (default: retrieval_eval_results.json)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = run_retrieval_evaluation(csv_path=args.csv, k=args.k)

    # Print human-readable report
    print_retrieval_report(results)

    # Save to JSON
    output_path = Path(args.output)
    output_data = {
        "config": {"k": args.k},
        "results": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
