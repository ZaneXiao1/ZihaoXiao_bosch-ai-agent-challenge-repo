# Bosch AI Agent Challenge — Zihao Xiao

A production-ready, multi-source RAG agent that answers questions about Electronic Control Unit (ECU) specifications across the ECU-700 and ECU-800 product lines.

## Project

See [me-engineering-assistant/](me-engineering-assistant/) for the full implementation.

### Architecture highlights

- **LangGraph ReAct agent** — LLM autonomously selects which retrieval tool(s) to call based on the query
- **Dual Chroma in-memory vector stores** — separate indices for ECU-700 and ECU-800 series, ensuring both are always represented in cross-series queries
- **OpenAI + Databricks model provider abstraction** — switch providers via a single environment variable, zero code changes
- **MLflow pyfunc wrapper** — `predict()` method with typed error responses, ready for Databricks Model Serving
- **Databricks Asset Bundle (DAB)** — `databricks.yml` deploys the agent as a versioned MLflow model via an automated job
- **LLM-as-judge evaluation framework** — GPT-4o scores answers on 5 dimensions; release gate blocks registration if quality thresholds are not met

### Evaluation results

| Dataset | Questions | Result |
|---|---|---|
| Official (keyword pass/fail) | 10 | **10/10 PASS** |
| LLM-as-judge (GPT-4o) | 10 | avg correctness 4.9/5, faithfulness 5.0/5 |
| Extended dataset | 60 | single-source, cross-series, out-of-scope, corner cases |

## Quick Start

```bash
cd me-engineering-assistant

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run evaluation
python scripts/run_evaluation.py
```

## Documentation

- [Full README & Architecture](me-engineering-assistant/README.md) — design decisions, deployment guide, testing strategy, monitoring
- [Evaluation Results](me-engineering-assistant/evaluation_results.md) — 10/10 run output with full answers
- [Challenge Brief](ai-engineering-coding-challenge.md)
