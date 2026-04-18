"""
Build resources and log the ME Engineering Assistant as an MLflow pyfunc model.

Intended to be the entry point for the Databricks Asset Bundle job: it runs
non-interactively, owns the MLflow run lifecycle, records parameters and
smoke-test metrics, and optionally registers the model in MLflow Model Registry.

Usage
-----
Local:
    python scripts/train_and_log_model.py

Databricks job (see ../databricks.yml):
    python scripts/train_and_log_model.py \\
        --experiment-name /Users/<user>/me-engineering-assistant \\
        --registered-model-name me_engineering_assistant
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import mlflow

# Ensure `src/` is importable when invoked as a plain script (e.g. Databricks
# job) without a prior `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from me_assistant.agent.graph import create_agent, query_agent  # noqa: E402
from me_assistant.config import MODEL_PROVIDER, validate_config  # noqa: E402
from me_assistant.documents.loader import load_documents  # noqa: E402
from me_assistant.evaluation.evaluator import (  # noqa: E402
    DIMENSIONS,
    run_evaluation,
    summarize_results,
)
from me_assistant.model.mlflow_wrapper import log_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_and_log_model")

# Default smoke-test questions: a focused subset of tests/test_queries.csv
# covering single-source, cross-series, and feature-availability categories.
_SMOKE_QUESTIONS = [
    "What is the maximum operating temperature for the ECU-750?",
    "How much RAM does the ECU-850 have?",
    "Which ECU models support Over-the-Air (OTA) updates?",
]

# Retriever k is set in me_assistant.documents.store.get_retriever default.
# Surfacing here so the value is explicitly logged as a run parameter.
_RETRIEVER_K = 3

# Offline eval datasets. The "official" set is the release gate; the
# "extended" set tracks coverage/generalisation and is advisory.
_TESTS_DIR = _REPO_ROOT / "tests"
_OFFLINE_EVAL_DATASETS: list[dict] = [
    {
        "name": "official",
        "csv": _TESTS_DIR / "test_queries.csv",
        "is_gate": True,
    },
    {
        "name": "extended",
        "csv": _TESTS_DIR / "test_queries_extended.csv",
        "is_gate": False,
    },
]

# Release gate: minimum per-dimension average on the official set.
# Failing any of these blocks model registration.
_GATE_MIN_DIM_AVG = 4.0
_GATE_DIMENSIONS = ("correctness", "faithfulness")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "me-engineering-assistant"),
        help="MLflow experiment name (default from MLFLOW_EXPERIMENT_NAME or 'me-engineering-assistant').",
    )
    parser.add_argument(
        "--registered-model-name",
        default=os.getenv("MLFLOW_REGISTERED_MODEL_NAME"),
        help="If set, register the logged model under this name in MLflow Model Registry.",
    )
    parser.add_argument(
        "--run-name",
        default="train_and_log",
        help="MLflow run name.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the live agent smoke test (useful for CI without API keys).",
    )
    parser.add_argument(
        "--skip-offline-eval",
        action="store_true",
        help="Skip the LLM-as-judge offline eval on golden datasets.",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Run the gate eval but do not fail the job if thresholds are missed.",
    )
    return parser.parse_args()


def _run_smoke_test(agent) -> dict[str, float]:
    """Run a few live queries and return aggregate metrics.

    Measures end-to-end latency and whether the agent produced a non-empty,
    non-error answer. Errors are counted as failures but do not abort the run.
    """
    latencies: list[float] = []
    successes = 0
    for q in _SMOKE_QUESTIONS:
        start = time.perf_counter()
        try:
            answer = query_agent(agent, q)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            ok = bool(answer) and not answer.lower().startswith(
                ("error", "configuration error", "llm error",
                 "retrieval error", "embedding error")
            )
            if ok:
                successes += 1
            logger.info("Smoke q=%r latency=%.2fs ok=%s", q, elapsed, ok)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Smoke query failed: %s: %s", type(exc).__name__, exc)
            latencies.append(time.perf_counter() - start)

    return {
        "smoke_test_count": float(len(_SMOKE_QUESTIONS)),
        "smoke_test_success_count": float(successes),
        "smoke_test_success_rate": successes / len(_SMOKE_QUESTIONS),
        "smoke_test_latency_avg_sec": sum(latencies) / len(latencies),
        "smoke_test_latency_max_sec": max(latencies),
    }


def _run_offline_eval(agent, dataset: dict) -> dict:
    """Run LLM-as-judge eval on one dataset inside a nested MLflow run.

    Tokens and cost are captured via ``get_openai_callback`` which sums usage
    across both the agent's LLM calls and the judge LLM. Non-OpenAI providers
    produce zero-valued counters (callback only observes OpenAI traffic); in
    that case we skip the cost metrics rather than logging misleading zeros.

    Returns the summary dict (with dimension averages) so the caller can
    apply release-gate thresholds.
    """
    from langchain_community.callbacks import get_openai_callback

    name = dataset["name"]
    csv_path = dataset["csv"]
    logger.info("Offline eval [%s]: %s", name, csv_path)

    with mlflow.start_run(run_name=f"offline_eval_{name}", nested=True):
        mlflow.set_tags({
            "eval_type": "offline",
            "dataset": name,
            "is_release_gate": str(dataset["is_gate"]).lower(),
        })
        mlflow.log_params({
            "dataset_name": name,
            "dataset_path": str(csv_path.relative_to(_REPO_ROOT)),
            "judge_model": "gpt-4o",
        })

        start = time.perf_counter()
        with get_openai_callback() as cb:
            results = run_evaluation(agent, csv_path=str(csv_path))
            summary = summarize_results(results)
        wall_seconds = time.perf_counter() - start

        # Core quality metrics: one per dimension + overall.
        metrics: dict[str, float] = {
            "dataset_size": float(summary["total_questions"]),
            "overall_avg_score": float(summary["overall_avg_score"]),
            "overall_max_score": float(summary["overall_max_score"]),
            "latency_avg_sec": float(summary["latency_avg"]),
            "latency_min_sec": float(summary["latency_min"]),
            "latency_max_sec": float(summary["latency_max"]),
            "wall_seconds": float(wall_seconds),
        }
        for dim, avg in summary["dimension_averages"].items():
            metrics[f"avg_{dim}"] = float(avg)

        # Token / cost telemetry (OpenAI only; zero for other providers).
        if cb.total_tokens > 0:
            metrics.update({
                "tokens_total": float(cb.total_tokens),
                "tokens_prompt": float(cb.prompt_tokens),
                "tokens_completion": float(cb.completion_tokens),
                "cost_usd": float(cb.total_cost),
                "cost_usd_per_query": float(cb.total_cost) / max(len(results), 1),
            })

        mlflow.log_metrics(metrics)

        # Persist full per-question results as an artifact for later review.
        out_path = _REPO_ROOT / f"eval_{name}_latest.json"
        out_path.write_text(
            json.dumps(
                {"summary": summary, "results": [r.to_dict() for r in results]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(out_path), artifact_path="offline_eval")

        logger.info(
            "[%s] overall=%.2f/%d | tokens=%d | cost=$%.4f | %.1fs",
            name,
            summary["overall_avg_score"],
            summary["overall_max_score"],
            cb.total_tokens,
            cb.total_cost,
            wall_seconds,
        )
        return summary


def _check_release_gate(summary: dict) -> list[str]:
    """Return a list of gate failure messages (empty list = pass)."""
    failures: list[str] = []
    for dim in _GATE_DIMENSIONS:
        avg = summary["dimension_averages"].get(dim, 0.0)
        if avg < _GATE_MIN_DIM_AVG:
            failures.append(
                f"{dim}={avg:.2f} below threshold {_GATE_MIN_DIM_AVG}"
            )
    return failures


def main() -> int:
    args = _parse_args()

    # Fail fast on missing env vars rather than mid-run.
    validate_config()

    mlflow.set_experiment(args.experiment_name)

    logger.info("Loading documents for chunk-count telemetry...")
    documents = load_documents()
    chunks_700 = sum(1 for d in documents if d.metadata["series"] == "ECU-700")
    chunks_800 = sum(1 for d in documents if d.metadata["series"] == "ECU-800")

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "project_name": "me-engineering-assistant",
            "project_version": "0.1.0",
            "model_provider": MODEL_PROVIDER,
            "llm_model_name": os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            "embedding_model_name": os.getenv(
                "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
            ),
            "vector_store_type": "chroma-in-memory",
            "chunking_strategy": "section-based-markdown",
            "retriever_k": _RETRIEVER_K,
            "agent_framework": "langgraph-react",
        })

        mlflow.log_metrics({
            "chunk_count_total": float(len(documents)),
            "chunk_count_ecu_700": float(chunks_700),
            "chunk_count_ecu_800": float(chunks_800),
        })

        mlflow.set_tags({
            "task": "rag-qa",
            "domain": "ecu-engineering",
        })

        agent = None
        if not args.skip_smoke_test:
            logger.info("Running smoke test against live agent...")
            agent = create_agent()
            smoke_metrics = _run_smoke_test(agent)
            mlflow.log_metrics(smoke_metrics)
        else:
            logger.info("Skipping smoke test (--skip-smoke-test).")

        gate_failures: list[str] = []
        if not args.skip_offline_eval:
            if agent is None:
                agent = create_agent()
            for dataset in _OFFLINE_EVAL_DATASETS:
                summary = _run_offline_eval(agent, dataset)
                if dataset["is_gate"]:
                    gate_failures = _check_release_gate(summary)
        else:
            logger.info("Skipping offline eval (--skip-offline-eval).")

        if gate_failures and not args.skip_gate:
            mlflow.set_tag("release_gate", "failed")
            raise RuntimeError(
                "Release gate failed on official dataset: "
                + "; ".join(gate_failures)
                + ". Re-run with --skip-gate to override."
            )
        if gate_failures:
            mlflow.set_tag("release_gate", "overridden")
            logger.warning("Gate failures overridden: %s", gate_failures)
        elif not args.skip_offline_eval:
            mlflow.set_tag("release_gate", "passed")

        logger.info("Logging pyfunc model to MLflow...")
        model_info = log_model(
            registered_model_name=args.registered_model_name,
        )

        logger.info("=" * 60)
        logger.info("Run ID       : %s", run.info.run_id)
        logger.info("Model URI    : %s", model_info.model_uri)
        if args.registered_model_name:
            logger.info("Registered as: %s", args.registered_model_name)
        logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
