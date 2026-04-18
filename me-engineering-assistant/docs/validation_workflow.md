# Validation Workflow

This document defines the standard testing and validation flow for the ME Engineering Assistant.
The goal is to make it obvious:

- when to run each check
- which script to run
- where results are recorded
- how to debug failures

## Overview

Use the workflow in four layers:

1. Local development checks
2. Functional acceptance
3. Answer-quality evaluation
4. Release validation and MLflow tracking

Each layer has a different purpose. Do not use one layer as a substitute for all the others.

## 1. Local Development Checks

Purpose:
- catch code regressions quickly
- validate error handling and retrieval plumbing
- avoid spending API cost during normal development

Command:

```bash
pytest tests/ -v -m "not integration"
```

What this covers:
- config validation
- wrapper input validation
- structured error handling
- document loading
- retrieval metadata and chunk integrity

When to run:
- after any code change
- before pushing a branch
- before running more expensive live evaluations

Where results are recorded:
- terminal output only

If this fails:
- fix code or configuration first
- do not proceed to live evaluation until this passes

## 2. Functional Acceptance

Purpose:
- verify that the agent can answer the core benchmark questions end-to-end
- confirm basic latency and pass/fail performance on the official dataset

Command:

```bash
python scripts/run_evaluation.py
```

What this covers:
- runs the 10-question official dataset in `tests/test_queries.csv`
- prints answer previews, sources queried, iteration count, and timing
- reports Tier 1 pass/fail status

When to run:
- after changes to prompts, tools, graph routing, or retrieval behavior
- before a live demo
- before full answer-quality evaluation

Where results are recorded:
- terminal output
- optionally summarized in `evaluation_results.md`

If this fails:
- first inspect whether the failure is retrieval-related or answer-generation-related
- then run retrieval diagnostics if needed

## 3. Retrieval Diagnostics

Purpose:
- inspect whether the retriever is surfacing the right chunks
- debug failures that may look like answer problems but are actually retrieval problems

Command:

```bash
python scripts/run_retrieval_eval.py
```

What this covers:
- direct chunk ranking inspection without the full agent loop
- retrieved source file, section, rank, and similarity score

When to run:
- when `run_evaluation.py` fails unexpectedly
- when a prompt change seems correct but answers still degrade
- when debugging cross-series comparison behavior

Where results are recorded:
- terminal output
- `retrieval_eval_results.json` if saved by the script

Failure analysis hint:
- if the wrong chunks are retrieved, fix retrieval before changing prompt logic

## 4. Answer-Quality Evaluation

Purpose:
- measure answer quality beyond keyword pass/fail
- track correctness, completeness, groundedness, relevance, and formatting

Command:

```bash
python scripts/run_eval.py
```

What this covers:
- LLM-as-judge scoring on the configured dataset
- per-question and aggregate metrics for:
  - correctness
  - completeness
  - faithfulness
  - relevance
  - format compliance

When to run:
- after prompt or agent-behavior changes
- when comparing model variants
- before preparing a final submission or release candidate

Where results are recorded:
- terminal summary
- `eval_results.json`

If this fails:
- low correctness usually indicates factual or synthesis problems
- low faithfulness usually indicates hallucination or retrieval grounding issues
- low format compliance usually indicates response-shape or citation issues

## 5. Release Validation And MLflow Tracking

Purpose:
- run the formal versioned validation pipeline
- record metrics and artifacts in MLflow
- apply a release gate before model registration

Recommended command:

```bash
python scripts/train_and_log_model.py --experiment-name my-local-experiment
```

Lower-cost local variant:

```bash
python scripts/train_and_log_model.py \
    --experiment-name my-local-experiment \
    --skip-offline-eval
```

What this covers:
- validates environment configuration
- logs project parameters to MLflow
- logs chunk-count telemetry
- runs smoke tests
- runs offline evaluation on official and extended datasets
- applies the release gate
- logs the MLflow pyfunc model

When to run:
- before deployment
- before final handoff or submission
- when you need a persistent experiment record

Where results are recorded:
- MLflow experiment runs
- local `mlflow.db`
- `mlruns/`
- offline evaluation artifacts written by the script

## Recommended Standard Order

For a normal feature change, use this order:

1. `pytest tests/ -v -m "not integration"`
2. `python scripts/run_evaluation.py`
3. `python scripts/run_eval.py`
4. `python scripts/train_and_log_model.py --experiment-name ...`

For a quick local development loop, stopping after step 1 or 2 is usually enough.

## Results Matrix

| Goal | Command | Main output location |
|---|---|---|
| Fast code validation | `pytest tests/ -v -m "not integration"` | Terminal |
| Core functional check | `python scripts/run_evaluation.py` | Terminal |
| Retrieval debugging | `python scripts/run_retrieval_eval.py` | Terminal + JSON |
| Answer-quality evaluation | `python scripts/run_eval.py` | Terminal + `eval_results.json` |
| Formal tracked release check | `python scripts/train_and_log_model.py ...` | MLflow + artifacts |

## Failure Triage

When a result looks wrong, debug in this order:

1. Configuration
- Check `.env`, provider settings, API keys, and installed dependencies.

2. Retrieval
- Run `python scripts/run_retrieval_eval.py`.
- If the wrong chunks are returned, fix retrieval first.

3. Answer generation
- If retrieval is correct but answers are still poor, inspect prompt behavior, tool routing, and answer formatting.

4. Evaluation dimensions
- Use the LLM-as-judge output to identify whether the main problem is correctness, faithfulness, completeness, or format compliance.

## Practical Rules

- Do not treat keyword pass/fail as the only quality signal.
- Do not skip retrieval diagnostics when answers degrade unexpectedly.
- Use MLflow-tracked runs for release decisions, not just ad hoc terminal checks.
- Keep local unit tests fast so they remain part of the daily workflow.

## Suggested Team Policy

If this project is extended, a sensible policy is:

- Every code change must pass `pytest tests/ -v -m "not integration"`.
- Any agent or prompt change should also pass `python scripts/run_evaluation.py`.
- Any release candidate should run through `python scripts/train_and_log_model.py` with the release gate enabled.
