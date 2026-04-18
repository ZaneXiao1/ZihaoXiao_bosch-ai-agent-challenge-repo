# Tier 2 Implementation Plan — Production & MLOps Excellence

## Context

Tier 1 is fully complete: 10/10 evaluation pass, pylint 10/10, working MLflow `log_model()` + `predict()`, proper Python package structure. Tier 2 requires production hardening: **Databricks Asset Bundle (DAB) packaging**, **automated deployment job**, **error handling**, and **documented testing/validation strategy**.

### Tier 2 Success Criteria (from challenge spec)

- DAB deploys successfully in provided workspace
- MLflow model serves predictions via REST API
- Error handling covers common failure modes

---

## Step 1: Custom Exceptions — `src/me_assistant/exceptions.py` (NEW)

Create a minimal exception hierarchy for structured error handling:

```
MEAssistantError (base)
├── ConfigurationError   — missing env vars, bad provider, import failures
├── EmbeddingError       — embedding API failures during indexing/retrieval
├── RetrievalError       — vector store failures
└── LLMError             — LLM API failures (auth, rate limit, timeout)
```

**Why:** Enables callers to catch specific failure modes and return actionable error messages instead of raw tracebacks. The `predict()` endpoint can return `"Configuration error: ..."` vs `"LLM error: ..."` so API consumers know what went wrong.

---

## Step 2: Error Handling — Enhance Existing Modules

### `src/me_assistant/config.py`

- Add `validate_config()` function — pre-checks required env vars for the configured provider, raises `ConfigurationError` with actionable message. This consolidates the preflight logic already in `run_evaluation.py` into a reusable function.
- Wrap `get_llm()` and `get_embeddings()` in try/except to catch `ImportError` (missing `langchain-databricks`) and auth errors, re-raising as `ConfigurationError`.
- Add `max_retries=2` to `ChatOpenAI()` constructor (built-in LangChain retry, no custom loop needed).

### `src/me_assistant/agent/tools.py`

- In `_retrieve_with_fallback`: log the exception class name at WARNING level for better observability. Behavior (fallback to full doc) stays the same — pure observability improvement.

### `src/me_assistant/agent/graph.py`

- `create_agent()`: wrap `build_vector_stores()` and `get_llm()` in try/except, re-raise as specific `MEAssistantError` subclasses.
- `query_agent()`: wrap `agent.invoke()`, catch LLM-specific exceptions (e.g. `openai.AuthenticationError`, `openai.RateLimitError`) and re-raise as `LLMError`.

### `src/me_assistant/model/mlflow_wrapper.py`

- `load_context()`: wrap `create_agent()` with try/except for `MEAssistantError`, log the error and re-raise (so MLflow serving knows the model failed to load).
- `predict()`: differentiate error types — return `"Configuration error: ..."` vs `"LLM error: ..."` vs `"Error: ..."` so API consumers can distinguish failure modes.

---

## Step 3: Fix `log_model()` — `src/me_assistant/model/mlflow_wrapper.py`

Current bugs/gaps to fix:

| Issue | Current | Fix |
|-------|---------|-----|
| Wrong pip dep | `faiss-cpu` (not used) | Replace with `chromadb`, `langchain-chroma`, `pydantic`, `python-dotenv` |
| Deprecated param | `artifact_path=` | Change to `name=` |
| Relative paths | `src/me_assistant/data/...` | Resolve to absolute via `Path(__file__)` |
| No flexibility | Hardcoded everything | Add optional `registered_model_name`, `tags` params |

Corrected `pip_requirements`:
```python
["langchain>=0.3.0", "langchain-openai>=0.2.0", "langchain-community>=0.3.0",
 "langchain-chroma>=0.1.0", "langgraph>=0.2.0", "chromadb>=0.5.0",
 "mlflow>=2.15.0", "pydantic>=2.0", "python-dotenv>=1.0.0"]
```

---

## Step 4: Monitoring Module — `src/me_assistant/monitoring.py` (NEW)

Lightweight module with two helpers:

- `timed_query(agent, question)` — runs query with timing, returns `{answer, latency_seconds, error}`
- `log_evaluation_metrics(accuracy, avg_latency, total_queries)` — logs aggregate metrics to active MLflow run

Keeps timing/metrics logic separate from business logic.

---

## Step 5: Enhance Evaluation Script — `scripts/run_evaluation.py`

- Refactor `_preflight_check()` to call the new `validate_config()` from config.py (DRY)
- Add optional MLflow logging via `LOG_TO_MLFLOW` env var or `--log-mlflow` CLI flag
- When enabled: start an MLflow run, log per-query latency + pass/fail as step metrics, log aggregate accuracy/latency as summary, log `evaluation_results.md` as artifact

---

## Step 6: Deployment Script — `scripts/deploy_model.py` (NEW)

The script that the DAB job will execute:

1. Call `validate_config()` — fail fast on missing env vars
2. Set MLflow tracking URI to `"databricks"` (auto-detect if running in Databricks, else local)
3. Set experiment path from `MLFLOW_EXPERIMENT_PATH` env var (default: `/Shared/me-engineering-assistant`)
4. Call enhanced `log_model()` with tags (`model_version`, `provider`)
5. Optionally register the model via `mlflow.register_model()` if `MLFLOW_MODEL_NAME` env var is set
6. Optionally run evaluation if `RUN_EVALUATION=true`
7. Print the model URI for the job log

---

## Step 7: DAB Configuration — `databricks.yml` (NEW)

```yaml
bundle:
  name: me-engineering-assistant

variables:
  model_name:
    default: me-engineering-assistant
  experiment_path:
    default: /Shared/me-engineering-assistant

artifacts:
  me_assistant_wheel:
    type: whl
    path: .

resources:
  jobs:
    deploy_model:
      name: "[${bundle.target}] ME Assistant - Deploy Model"
      tasks:
        - task_key: deploy
          spark_python_task:
            python_file: scripts/deploy_model.py
          libraries:
            - whl: ../artifacts/me_assistant_wheel/*.whl

targets:
  dev:
    mode: development
    default: true
  staging: {}
  prod: {}
```

**Key decision:** Use `spark_python_task` (simple, no entry points needed) rather than `python_wheel_task` (would require adding `[project.scripts]` to pyproject.toml).

---

## Step 8: Error Handling Tests — `tests/test_error_handling.py` (NEW)

Unit tests using mocking (no API key required):

- `test_validate_config_missing_openai_key` — verify `ConfigurationError` raised
- `test_validate_config_missing_databricks_vars` — verify `ConfigurationError` raised
- `test_validate_config_valid` — verify no exception for correct config
- `test_predict_returns_structured_error` — mock `create_agent` to fail, verify error message format

---

## Step 9: Update README.md

Add three new sections:

### "Deployment" section
- DAB commands: `databricks bundle deploy`, `databricks bundle run`
- Environment variable configuration for Databricks
- Model serving setup

### "Testing & Validation Strategy" section
- **Test pyramid:** unit (no API) -> integration (API) -> end-to-end evaluation (10 queries)
- **Golden dataset:** the 10-question `test_queries.csv` with keyword-based pass/fail
- **Proposed evaluation metrics:** factual accuracy, response latency (P50/P95), source attribution rate, hallucination rate
- **Production monitoring:** MLflow metric logging, alerting thresholds, weekly evaluation runs
- **Domain expertise validation:** how to extend golden dataset with SME review

### "Performance Monitoring" section
- What metrics are tracked (latency, accuracy, error rate)
- Where to find them (MLflow experiment dashboard)
- Alerting strategy

### Update Project Structure tree
Add new files: `exceptions.py`, `monitoring.py`, `deploy_model.py`, `test_error_handling.py`, `databricks.yml`

---

## Execution Order

| # | File | Action | Depends On |
|---|------|--------|------------|
| 1 | `src/me_assistant/exceptions.py` | Create | — |
| 2 | `src/me_assistant/config.py` | Modify | Step 1 |
| 3 | `src/me_assistant/agent/tools.py` | Modify | Step 1 |
| 4 | `src/me_assistant/agent/graph.py` | Modify | Step 1 |
| 5 | `src/me_assistant/model/mlflow_wrapper.py` | Modify | Steps 1-4 |
| 6 | `src/me_assistant/monitoring.py` | Create | — |
| 7 | `scripts/run_evaluation.py` | Modify | Steps 2, 6 |
| 8 | `scripts/deploy_model.py` | Create | Steps 2, 5 |
| 9 | `databricks.yml` | Create | — |
| 10 | `tests/test_error_handling.py` | Create | Steps 1-4 |
| 11 | `README.md` | Modify | All above |

---

## Verification Checklist

- [ ] `pylint src/me_assistant` — must remain > 85% (currently 10/10)
- [ ] `pytest tests/ -v -m "not integration"` — must pass (includes new error handling tests)
- [ ] `pytest tests/ -v` — must pass (requires API key)
- [ ] `python scripts/run_evaluation.py` — must still achieve 10/10
- [ ] `python scripts/deploy_model.py` — runs locally, logs model to local MLflow
- [ ] `databricks bundle validate` — passes (if Databricks CLI available)

---

## New Files Summary

| File | Purpose |
|------|---------|
| `src/me_assistant/exceptions.py` | 4 custom exception classes |
| `src/me_assistant/monitoring.py` | Timing + MLflow metric helpers |
| `scripts/deploy_model.py` | Deployment script for DAB job |
| `tests/test_error_handling.py` | Unit tests for error paths |
| `databricks.yml` | Databricks Asset Bundle config |

## Modified Files Summary

| File | Changes |
|------|---------|
| `src/me_assistant/config.py` | Add `validate_config()`, error wrapping, `max_retries` |
| `src/me_assistant/agent/tools.py` | Better exception logging in fallback |
| `src/me_assistant/agent/graph.py` | Error handling in `create_agent()` + `query_agent()` |
| `src/me_assistant/model/mlflow_wrapper.py` | Fix pip deps, flexible `log_model()`, typed errors in `predict()` |
| `scripts/run_evaluation.py` | Use `validate_config()`, optional MLflow logging |
| `README.md` | Add Deployment, Testing Strategy, Monitoring sections + update structure |
