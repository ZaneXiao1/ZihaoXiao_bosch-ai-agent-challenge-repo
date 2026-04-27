# Quick Start Guide

Get the ME Engineering Assistant running locally and see evaluation results in under 5 minutes.

---

## 1. Install

```bash
cd me-engineering-assistant
pip install -e ".[dev]"
```

All commands below assume you are inside the `me-engineering-assistant/` directory.

## 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

## 3. Chat with the Agent

```bash
python scripts/chat_cli.py
```

Try these example queries:

```
You> What is the maximum operating temperature for the ECU-750?
You> Compare the CAN bus capabilities of ECU-750 and ECU-850.
You> Does the ECU-750 support OTA updates?
You> What is the price of the ECU-850?
```

Type `exit` to quit.

---

## 4. Run Evaluations

### Unit Tests (no API key needed)

```bash
pytest tests/ -v -m "not integration"
```

Files: `tests/test_*.py` (error handling, retrieval, state management). Use before committing changes to verify no regressions in core logic.

### Integration Tests (requires API key)

```bash
pytest tests/ -v
```

Runs the full test suite, including live agent and retrieval checks that require `OPENAI_API_KEY`. Target coverage: >= 85%.

### Tier 1: Keyword Pass/Fail (Release Gate)

```bash
python scripts/run_evaluation.py
```

Dataset: `tests/test_queries.csv` (10 official questions). Method: keyword matching against reference answers. Pass criteria: >= 8/10 correct. Cost: free (no judge LLM calls). Speed: < 2 seconds. Use before merging to `main` as a lightweight continuous-integration gate.

Output: binary pass/fail report with per-question breakdown.

### Tier 3: LLM-as-Judge (60 questions)

```bash
python scripts/run_eval.py
```

By default this runs the 60-question extended set `tests/test_queries_extended.csv`.
You can also switch datasets explicitly:

```bash
python scripts/run_eval.py --dataset official
python scripts/run_eval.py --dataset extended
python scripts/run_eval.py --csv tests/test_queries_extended.csv
```

Dataset: `tests/test_queries_extended.csv` (60 extended questions). Method: GPT-4o judge scoring across 5 dimensions: correctness, completeness, faithfulness, relevance, and format compliance.

Use before pre-production deployment or major prompt/retrieval changes. Results are saved to `eval_results.json` with per-question scores, retrieved context, model answers, latency, and judge rationales.

### Full Pipeline: Train + Evaluate + Log to MLflow

```bash
python scripts/train_and_log_model.py
```

Runs smoke tests, both evaluation tiers, and logs offline evaluation metrics to MLflow.

Tracked in MLflow:
- Chunk counts
- Smoke test results (success rate, latency)
- Quality scores (Tier 3 dimension averages)

View results:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# Open http://localhost:5001
```

### LangSmith (Optional): Trace & Debug Agent Calls

[LangSmith](https://smith.langchain.com) provides detailed tracing of every LLM call, retrieval step, and chain execution.

Key signals in LangSmith:
- Cost metrics (tokens, USD per query)
- Request volume — detect usage patterns and anomalies
- Latency — P50/P95/P99 latencies; alert on SLA breaches
- Error traces — API failures, retrieval errors, timeout patterns

1. Sign up at https://smith.langchain.com and create an API key in **Settings**.
2. Add these variables to your `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-your-key-here
LANGCHAIN_PROJECT=me-engineering-assistant
```

3. Re-run any evaluation or chat session — traces will automatically appear in your LangSmith dashboard at https://smith.langchain.com.
