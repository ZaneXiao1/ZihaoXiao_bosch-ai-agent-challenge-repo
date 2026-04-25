# ME Engineering Assistant

A production-ready Retrieval-Augmented Generation (RAG) agent that answers technical questions about Electronic Control Unit (ECU) specifications across the ECU-700 and ECU-800 product lines. Built with LangGraph, LangChain.

---

## Table of Contents

1. [Architectural Design](#1-architectural-design)
2. [Setup & Deployment](#2-setup--deployment)
3. [Testing & Validation Strategy](#3-testing--validation-strategy)
4. [Limitations & Future Work](#4-limitations--future-work)

---

For a full directory listing with file descriptions, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

---

# 1. Architectural Design

## 1.1 System Overview

The ME Engineering Assistant is a three-layer RAG architecture:

```
                     ┌───────────────────────────┐
                     │  User Query               │
                     │  (Natural Language)       │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────┐
               │         Layer 1: LLM Reasoning        │
               │  Agent decides which tool(s) to call  │
               └───────────┬───────────────┬───────────┘
                           │               │
                           ▼               ▼
               ┌───────────────┐ ┌─────────────────┐
               │  Layer 2:     │ │  Layer 2:       │
               │  ECU-700      │ │  ECU-800        │
               │  Vector Store │ │  Vector Store   │
               │  (Chroma)     │ │  (Chroma)       │
               └───────┬───────┘ └────────┬────────┘
                       │                  │
                       └────────┬─────────┘
                                │
                                ▼
               ┌───────────────────────────────────────┐
               │       Layer 3: Answer Generation      │
               │  LLM synthesizes answer from chunks   │
               │  with source citations                │
               └───────────────────────────────────────┘
```

The system handles three key responsibilities:
1. **LLM reasoning** — deciding which retrieval tools to invoke
2. **Semantic search** — retrieving relevant chunks from vector stores
3. **Answer generation** — producing grounded, cited responses

### Prompt Design

The agent's behavior is governed by prompts at three levels:

**System Prompt** (`agent/prompts.py`) — injected once at the start of conversation, defines 10 rules across three categories:

- **Retrieval discipline (Rules 1–3):** Always search before answering; single-series queries search one store, comparisons search both. Prevents the LLM from answering "from memory" without grounding.
- **Anti-hallucination (Rules 6–7):** If a value is not in the retrieved text, say so explicitly. Includes a self-check instruction: "mentally locate the exact sentence before stating any value." Uses exact terminology from docs (e.g., "Yocto-based Linux OS" not "Linux-based OS").
- **Answer discipline (Rules 4, 8–10):** Answer only what was asked, match response length to question complexity, use prose only (no tables/bullet lists), and include few-shot examples of well-scoped answers.

**Tool-level prompts** (`agent/tools.py`) — each tool's return value is prefixed with a guardrail reminder:
```
[IMPORTANT: Only the attributes explicitly mentioned below exist
in this documentation. If the answer is not found, the information
is NOT available.]
```
This reinforces anti-hallucination at the point where the LLM reads retrieved context, not just in the system prompt.

**Force-answer prompt** (`agent/nodes.py`) — injected as a late SystemMessage when the iteration ceiling is reached. Instructs the LLM to answer with what it already has and repeats the anti-fabrication constraint, ensuring the safety-ceiling path does not degrade answer quality.

## 1.2 Agent Graph Architecture

The agent is implemented as a custom LangGraph `StateGraph` with three named nodes and explicit state management. This replaces the prebuilt `create_react_agent` black box with a production-grade, observable graph.

**Key design principle:** Agent autonomously decides when to stop (by not emitting tool_calls), while the graph enforces a deterministic 3-iteration safety ceiling via routing logic.

### Graph Flow Diagram

```
                        ┌──────────────────────┐
                        │    User Question     │
                        └──────────┬───────────┘
                                   │
                                   ▼
                 ┌─────────────────────────────────────┐
            ┌───▶│          AGENT NODE                 │
            │    │  LLM with tools bound               │
            │    │  Reads full conversation history    │
            │    │  Decides: call tools or answer?     │
            │    └──────┬──────────┬──────────┬────────┘
            │           │          │          │
            │     No tool_calls   Has calls  Has calls
            │     (answer ready)  iter < 3   iter >= 3
            │           │          │          │
            │           ▼          │          ▼
            │     ┌──────────┐     │   ┌──────────────┐
            │     │   END    │     │   │ FORCE_ANSWER │
            │     │ (return  │     │   │ LLM without  │
            │     │  answer) │     │   │ tools bound  │
            │     └──────────┘     │   │ (must answer │
            │                      │   │  with what   │
            │                      │   │  it has)     │
            │                      │   └──────┬───────┘
            │                      ▼          │
            │              ┌─────────────┐    ▼
            │              │ TOOLS NODE  │  ┌──────────┐
            │              │ Execute     │  │   END    │
            │              │ retrieval   │  │ (return  │
            │              │ iter_count++│  │  answer) │
            └──────────────┤ Return      │  └──────────┘
              loop back    │ chunks      │
              to agent     └─────────────┘
```

### State Schema

| Field | Type | Purpose |
|---|---|---|
| `messages` | `Annotated[list, add_messages]` | Conversation history (append-only reducer) |
| `iteration_count` | `int` | Retrieval rounds executed (max: 3) |
| `sources_queried` | `list[str]` | Tools that fired (for observability) |

### Node Responsibilities

| Node | Role |
|---|---|
| **agent** | LLM reads tool docstrings and decides: call tools or answer? |
| **tools** | Execute retrieval, wrap results, increment iteration_count |
| **force_answer** | LLM without tools (structurally prevents further calls) |



### How Agent Decides When to Stop

The loop is driven by **Agent's judgment** with **graph-enforced safety ceiling**:

```
Iteration 1                Iteration 2                Iteration 3 (max)
─────────────────────    ─────────────────────    ─────────────────────
AGENT: need more info?   AGENT: need more info?   AGENT: still wants tools?
  → Yes, call tools        → Yes, call tools        → Blocked by ceiling
  → TOOLS: retrieve        → TOOLS: retrieve        → FORCE_ANSWER
  → iter_count = 1         → iter_count = 2         → Answer with what
  → Back to AGENT ──────▶  → Back to AGENT ──────▶    it already has
                                                     → END

At ANY iteration, if AGENT decides it has enough info:
  → No tool_calls emitted → END (return answer immediately)
```

### Why This Architecture

1. **Agent-driven iteration** — The Agent itself decides when it has enough information. The graph doesn't force tool calls or prevent legitimate answers.
   
2. **Explicit decision points** — Every iteration starts with agent_node, which sees the full conversation and can make an informed "continue or stop" decision
   
3. **Deterministic safety ceiling** — The 3-iteration maximum is enforced via routing logic (not LLM hope). Even if the Agent "wants" more tools at iteration 3, the graph routes to `force_answer` instead
   
4. **Parallel tool optimization** — Multiple tools can execute in one iteration (counting as 1 retrieval round). This reduces latency for cross-series queries while still updating iteration_count correctly
   
5. **Structural termination guarantee** — The `force_answer` node uses a plain LLM without `bind_tools()`, making it structurally impossible for tool_calls to be generated
   
6. **Complete audit trail** — The `add_messages` reducer preserves every turn: human input → agent decision → tool result → agent re-evaluation → final answer. Perfect for debugging and validation
   
7. **Observable per-node metrics** — Each node (agent, tools, force_answer) is a separate LangSmith span, enabling precise latency, token, and error tracking by component

---

## 1.3 Retrieval Architecture

### Retrieval Tools

The agent has two tools, each bound to a dedicated vector store:

| Tool | Covers | When Used |
|---|---|---|
| `search_ecu_700_docs` | ECU-700 series (ECU-750) | Single-series queries about ECU-750, or one side of a cross-series comparison |
| `search_ecu_800_docs` | ECU-800 series (ECU-850, ECU-850b) | Single-series queries about ECU-850/850b, or one side of a cross-series comparison |

The agent reads the tool docstrings and autonomously decides which tool(s) to call. For cross-series comparisons it calls both tools in a single iteration.

### Two-Layer Multi-Source Design

**Layer 1 (LLM Reasoning):**
- The agent LLM reads tool docstrings and autonomously decides which tool(s) to call
- Single tool: focused, single-series queries (e.g., "What is the max temp for ECU-750?")
- Both tools: cross-series comparisons (e.g., "Compare OTA capabilities across all models")
- Pure language understanding, no embeddings involved

**Layer 2 (Vector Search):**
- Each tool internally performs cosine similarity search against its dedicated Chroma vector store
- Returns the top-k most semantically relevant chunks
- Falls back to full document if retrieval returns < 200 characters
- Pure vector math, no LLM involved

### Dual-Index Strategy

The system maintains **two independent Chroma in-memory vector stores** — one per ECU series:

```python
vector_stores = {
    "ecu_700": Chroma(collection_name="ecu-700", embedding_function=embedder),
    "ecu_800": Chroma(collection_name="ecu-800", embedding_function=embedder),
}
```

**Why separate indices:**
- A single combined index would allow high-similarity ECU-800 chunks (e.g., "OTA Update Capability") to crowd out low-scoring but relevant ECU-700 chunks (e.g., "OTA updates are not supported")
- Separate indices guarantee both series are represented when the agent queries across the product line
- For a corpus of ~11 chunks, the memory overhead is negligible

### Section-Based Chunking

Documents are split on **natural section boundaries** rather than fixed token counts:

- **ECU-700:** Bold numbered sections (`**1. Introduction**`, `**2. Specifications**`, etc.)
- **ECU-800:** Markdown level-2 headings (`## Overview`, `## Technical Specs`, etc.)

**Why not fixed-size chunks:**
- ECU spec documents contain tables that must remain intact
- Mid-table splits cause embedding models to miss key values
- Section-based splits preserve table structure and provide full context per chunk
- With ~11 sections across three documents, granularity is ideal

### Context Prefix Injection

Every chunk's `page_content` is prefixed with its series and model metadata:

```
[ECU-700 Series | ECU-750] Diagnostics
The ECU-750 provides basic diagnostic trouble codes (DTCs)…
```

**Why injection into page_content:**
- Embedding models only see `page_content`, not metadata fields
- Without this prefix, a chunk like "runs a Yocto-based Linux OS" carries no signal about which ECU it belongs to
- The prefix makes every chunk self-contained and self-describing to the embedding model

### Embedding Model Selection: `text-embedding-3-small`

The embedding model is set to `text-embedding-3-small` (1,536 dimensions) rather than `text-embedding-3-large` (3,072 dimensions):

- Corpus is ~11 short chunks (~6 KB total)
- Small model provides sufficient semantic discrimination at this scale
- Large model adds latency and cost with no measurable accuracy benefit
- **Principle:** Scale model capacity to corpus size

---

## 1.4 Error Handling

The agent defines five custom exception types for clear error classification:

```python
# exceptions.py
MEAssistantError (base)
├── ConfigurationError        # Invalid provider config, missing API keys
├── EmbeddingError            # Vector store creation failed
├── RetrievalError            # Tool invocation or chunk retrieval failed
├── LLMError                  # LLM API auth, rate limit, timeout, or malformed response
└── DocumentLoadError         # Document loading or chunking failed
```

Each exception:
- Inherits from a common base for unified error handling
- Includes a descriptive message
- Is caught at the agent boundary and converted to a structured error response


---

# 2. Setup & Deployment

## 2.1 Deployment Context

**Note on Databricks Access:** This challenge does not provide Databricks workspace credentials. Following official guidance, this solution implements **manual MLflow logging with detailed deployment plan** as the deployment approach.

**What's Ready:**
- **MLflow pyfunc model** (`src/me_assistant/model/mlflow_wrapper.py`) — implements required `predict()` interface
- **Manual logging script** (`scripts/train_and_log_model.py`) — executes evaluation and logs metrics to MLflow
- **Detailed deployment specification** (Sections 2.2–2.3) — complete DAB configuration and deployment steps

Upon receiving Databricks workspace access, the solution deploys without code changes following the DAB steps in Section 2.2.

## 2.2 Quick Start (Local Development)

For local installation, configuration, and running the agent, see [QUICKSTART.md](QUICKSTART.md).

---

## 2.3 Deployment on Databricks with Asset Bundle

### Prerequisites

1. **Install Databricks CLI and authenticate:**
   ```bash
   pip install databricks-cli
   databricks configure --token
   # Follow prompts to enter workspace URL and personal access token
   ```

2. **Create a Databricks secret scope for credentials:**
   ```bash
   databricks secrets create-scope --scope me-engineering-assistant
   
   # Option A: Store OpenAI API key
   databricks secrets put --scope me-engineering-assistant --key openai_api_key
   # When prompted, paste your OpenAI API key
   
   # Option B: Use Databricks-hosted models (no external API key needed)
   # Databricks handles model serving via Foundation Model APIs
   ```

### Databricks Asset Bundle (DAB) Configuration

The `databricks.yml` defines two deployment targets:

```yaml
# databricks.yml (simplified — see file for full config)
artifacts:
  me_assistant_wheel:
    type: whl
    build: python -m build --wheel
    path: .

resources:
  jobs:
    train_and_log_model:
      tasks:
        - task_key: train_and_log
          job_cluster_key: job_cluster
          spark_python_task:
            python_file: ./scripts/train_and_log_model.py
            parameters:
              - --experiment-name
              - ${var.experiment_name}
              - --registered-model-name
              - ${var.registered_model_name}
          libraries:
            - whl: ./dist/*.whl
      job_clusters:
        - job_cluster_key: job_cluster
          new_cluster:
            spark_version: 15.4.x-scala2.12
            spark_env_vars:
              MODEL_PROVIDER: ${var.model_provider}
              OPENAI_API_KEY: "{{secrets/${var.secret_scope}/openai_api_key}}"

targets:
  dev:
    mode: development
    default: true
  prod:
    mode: production
    variables:
      experiment_name: /Shared/me-engineering-assistant-prod
```

### Deploy & Run

```bash
# Validate bundle configuration
databricks bundle validate -t dev

# Deploy (uploads wheel + notebooks to workspace)
databricks bundle deploy -t dev

# Run the training + logging job (includes evaluation)
databricks bundle run -t dev train_and_log_model

# For production
databricks bundle deploy -t prod
databricks bundle run -t prod train_and_log_model
```

### What the Deployment Job Does

The `train_and_log_model.py` job:

1. **Validates configuration** — checks all required environment variables
2. **Builds vector stores** — loads documents, creates embeddings, logs chunk counts to MLflow
3. **Runs smoke tests** — executes 3 live queries, logs latency and success rate
4. **Runs offline evaluation** — executes LLM-as-judge on official (10Q) and extended (60Q) datasets
5. **Applies release gate** — blocks model registration if `correctness < 4.0` or `faithfulness < 4.0` on official set
6. **Logs to MLflow Model Registry** — registers `MEAssistantModel` pyfunc if gate passes

All metrics, model artifacts, and evaluation logs are stored in MLflow under `/Shared/me-engineering-assistant`.

---

## 2.4 Deployment Readiness (Without Databricks Access)

**Context:** This challenge does not provide Databricks workspace credentials. Following the official guidance, this solution uses **manual MLflow logging with detailed deployment plan** as the fallback approach.

**What's Implemented:**
- **MLflow pyfunc model** (`src/me_assistant/model/mlflow_wrapper.py`) — implements required `predict()` interface for logging and serving
- **Manual logging script** (`scripts/train_and_log_model.py`) — can be executed locally or on any Python environment to log metrics and model artifacts
- **Detailed deployment plan** (Sections 2.1–2.3) — fully specifies how to deploy to Databricks with DAB configuration

Upon receiving Databricks workspace access, deployment would follow the steps outlined in Section 2.3 without any code changes.

---

## 2.5 Serving via REST API (Post-Deployment)

After model registration, enable serving in the Databricks UI:

1. Navigate to **Experiments > me-engineering-assistant > Latest Model**
2. Click **Enable Serving**

The model endpoint accepts:

```bash
curl -X POST \
  https://your-databricks-workspace/serving/models/me-assistant-model/invocations \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"question": "What is the max operating temperature for the ECU-750?"}
    ]
  }'
```

Expected response:
```json
{
  "predictions": [
    "The ECU-750 operates between -40°C and +85°C according to the ECU-700 Series Manual. Sources: ECU-700 Series Manual (Specifications section)."
  ]
}
```

The MLflow pyfunc wrapper handles:
- Input validation and type coercion
- Error handling with structured error messages prefixed by category
- Both DataFrame and dict inputs
- Consistent response formatting

---

## 2.6 Environment Variables

| Variable | Default | Description | When Required |
|---|---|---|---|
| `MODEL_PROVIDER` | `openai` | `openai` or `databricks` | Always |
| `OPENAI_API_KEY` | — | OpenAI API key | When `MODEL_PROVIDER=openai` |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | LLM model or endpoint name | Always |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Embedding model or endpoint | Always |
| `DATABRICKS_HOST` | — | Databricks workspace URL | When `MODEL_PROVIDER=databricks` |
| `DATABRICKS_TOKEN` | — | Databricks personal access token | When `MODEL_PROVIDER=databricks` |

---

# 3. Testing & Validation Strategy

## 3.1 Conceptual Framework

The evaluation strategy spans three stages:

**Offline Testing (Development)**
- Unit tests — error handling, retrieval logic, state management
- Integration tests — live agent queries with real LLM and embeddings

**Pre-Deployment Testing**
- Tier 1: Keyword pass/fail on official 10-question set (release gate)
- Tier 3: LLM-as-judge quality scoring on 60-question extended set
- Smoke tests: API connectivity and latency validation before full eval

**Online Monitoring (Production)**
- LangSmith: Request volume, latency (P50/P95/P99), error traces
- MLflow: Offline evaluation results, version-level metrics, experiment tracking

---

## 3.2 Evaluation Metrics

### Build & Indexing Metrics

| Metric | Purpose |
|---|---|
| `chunk_count_total` | Total chunks across both vector stores |
| `chunk_count_ecu_700` | Chunks from ECU-700 series |
| `chunk_count_ecu_800` | Chunks from ECU-800 series |

### Quality Metrics (LLM-as-Judge)

| Dimension | Scale | Measures |
|---|---|---|
| **Correctness** | 1–5 | Factual accuracy vs. reference answer |
| **Completeness** | 1–5 | Coverage of all required information |
| **Faithfulness** | 1–5 | Grounding in retrieved context (no hallucination) |
| **Relevance** | 1–5 | Direct relevance to the question |
| **Format Compliance** | 1–5 | Proper source attribution and formatting |

### Performance Metrics

| Metric | Purpose |
|---|---|
| `smoke_test_success_rate` | Fraction of smoke queries returning answers (no errors) |
| `smoke_test_latency_avg_sec` | Average end-to-end latency across smoke tests |
| `smoke_test_latency_max_sec` | Worst-case latency (P100) |
| `latency_avg_sec` | P50 latency across full eval set |
| `latency_max_sec` | P100 latency across full eval set |

### Cost Metrics

| Metric | Purpose |
|---|---|
| `tokens_total` | Total API tokens consumed in evaluation run |
| `cost_usd` | Estimated cost of the evaluation run |
| `cost_usd_per_query` | Average cost per query |

---

## 3.3 Evaluation Datasets

### Official Dataset (Tier 1 Release Gate)

| File | Size | Purpose |
|---|---|---|
| `tests/test_queries.csv` | 10 questions | Official standard test set; binary pass/fail gate |

This is the official evaluation dataset provided with the challenge, used to validate minimal correctness requirements.

### Extended Dataset

| File | Size | Purpose |
|---|---|---|
| `tests/test_queries_extended.csv` | 60 questions | Comprehensive quality assessment across diverse query domains |

Generated by LLM to mimic the official 10-question standard test set, the extended dataset covers:
- **Single-series queries** — e.g., "What are the I/O options for the ECU-750?"
- **Cross-series comparisons** — e.g., "Which ECU models support CAN with redundancy?"
- **Out-of-scope rejection** — e.g., "What is the cost of the ECU-850?"
- **Edge cases** — e.g., "Is there a model between ECU-750 and ECU-850?"

---

## 3.4 Automated Testing & Scripts

### Unit & Integration Tests (Local Development)

```bash
# Unit tests only (no API key required)
pytest tests/ -v -m "not integration"

# Full test suite (requires OPENAI_API_KEY)
pytest tests/ -v
```

**Files:** `tests/test_*.py` (error handling, retrieval, state management)  
**When to use:** Before committing changes; verify no regressions in core logic  
**Evaluation metrics:** Code coverage ≥ 85%

### Tier 1: Keyword Pass/Fail Evaluation (Release Gate)

```bash
python scripts/run_evaluation.py
```

**Dataset:** `tests/test_queries.csv` (10 official questions)  
**Method:** Keyword matching against reference answers  
**Pass criteria:** ≥ 8/10 correct  
**Cost:** Free (no API calls)  
**Speed:** < 2 seconds  
**When to use:** Before merging to main; continuous integration gate  
**Output:** Binary pass/fail report with per-question breakdown

### Tier 3: LLM-as-Judge Quality Evaluation

```bash
python scripts/run_eval.py
```

By default this runs the 60-question extended dataset. You can also choose:

```bash
python scripts/run_eval.py --dataset official
python scripts/run_eval.py --dataset extended
python scripts/run_eval.py --csv tests/test_queries_extended.csv
```

**Dataset:** `tests/test_queries_extended.csv` (60 extended questions)  
**Method:** GPT-4o judge scoring across 5 dimensions (correctness, completeness, faithfulness, relevance, format compliance)  
**Release gate logic** (in `train_and_log_model.py`):
```python
if avg_correctness < 4.0 or avg_faithfulness < 4.0:
    raise GateFailure("Model quality below threshold")
# Otherwise, register model version
```

**Cost:** ~$0.05–$0.10 per run  
**Duration:** ~2–3 minutes  
**When to use:** Before pre-production deployment; full release validation  
**Output:** Per-question scores, dimension averages, dimension-level pass/fail

### Smoke Tests (Pre-Deployment Validation)

**When:** Runs at the start of every evaluation in `train_and_log_model.py`  
**What:** 3 live queries validate API connectivity and measure latency  
**Metrics collected:** `smoke_test_success_rate`, `smoke_test_latency_avg_sec`, `smoke_test_latency_max_sec`  
**Purpose:** Early detection of API failures or SLA breaches before expensive full evals

---

## 3.5 Production Monitoring & Continuous Validation

### MLflow: Offline Evaluation Tracking

All offline evaluation results are logged to MLflow for version control and experiment tracking:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 and navigate to "me-engineering-assistant" experiment
```

**Tracked metrics:**
- Build & indexing stats (chunk counts)
- Smoke test results (success rate, latency)
- Quality scores (Tier 3 dimension averages)
- Cost metrics (tokens, USD per query)

**Purpose:** Version-level metrics for comparing model iterations; gating decisions recorded in MLflow

### LangSmith: Real-Time Production Monitoring

Monitor live agent behavior in production via LangSmith dashboard:

**Key signals:**
- **Request volume** — Detect usage patterns and anomalies
- **Latency** — P50/P95/P99 latencies; alert on SLA breaches
- **Error traces** — API failures, retrieval errors, timeout patterns

**Integration:** Agent queries are automatically traced to LangSmith if `LANGSMITH_API_KEY` is set

### Alerting Strategy (Optional)

Set up alerts on:
- **Quality:** `avg_correctness < 4.0` or `avg_faithfulness < 4.0` — model quality degradation (check MLflow)
- **Availability:** `smoke_test_success_rate < 1.0` — service/API failures (check LangSmith error traces)
- **Performance:** `smoke_test_latency_max_sec > 10` — SLA breach (check LangSmith latency dashboard)
- **Cost:** `cost_usd_per_query > threshold` — budget exceeded (check MLflow metrics)

---

# 4. Limitations & Future Work

## 4.1 Current Limitations

### RAG & Retrieval

**Static corpus with no update mechanism.** Documents are loaded from hardcoded markdown files at startup. Adding a new ECU series or updating an existing spec requires code changes and full redeployment — there is no incremental indexing or hot-reload capability.

**In-memory vector stores.** Chroma runs in-memory, which works for ~11 chunks but does not scale. Beyond ~10K chunks, memory pressure and latency degrade. A persistent vector database (pgvector, Pinecone, etc.) would be required for a production-scale corpus.

**Fixed retrieval k=3.** Every query retrieves exactly 3 chunks per store regardless of question complexity. A simple factoid question wastes context on irrelevant chunks; a complex cross-series comparison may miss relevant ones.

**Repeated retrieval returns identical results.** Multi-round retrieval against the same store with the same query returns the same chunks. There is no query rewriting, chunk deduplication, or k-expansion between iterations, so repeated rounds on the same store add no new information.

**Variable chunk granularity.** Section-based chunking produces chunks ranging from short preambles to large tables. Section boundaries don't always align with optimal retrieval granularity, and some chunks carry too little signal for the embedding model.

### Agent Framework

**Hard-coded iteration ceiling.** `MAX_ITERATIONS=3` is a fixed constant. Complex queries that legitimately need more retrieval rounds are forced to answer early via `force_answer_node`, potentially degrading answer quality.

**No error recovery in tool execution.** If a tool call fails (API timeout, empty retrieval), the error string is returned to the LLM as a `ToolMessage`. The agent has no explicit mechanism to retry with a different query, switch tools, or gracefully degrade — it simply sees the error and improvises.

**No session memory.** Each query starts with a fresh state (`iteration_count=0`, empty `messages`). The agent cannot reference prior questions in a session, so multi-turn workflows like "what about the ECU-850?" after discussing ECU-750 require full context re-specification.

**Soft guardrails for out-of-scope queries.** Rejection of irrelevant questions relies entirely on the LLM following SYSTEM_PROMPT Rule 6. There is no hard classifier or input filter — a sufficiently adversarial prompt could bypass the instruction.

### Cost & Operations

**No query caching.** Every query re-embeds and re-searches the vector store, even for identical questions asked minutes apart. At scale, this wastes the majority of embedding API calls on repeated queries.

**No cost budget or circuit breaker.** Token usage is logged to MLflow after evaluation runs, but there is no per-query cost tracking or budget enforcement at runtime. A burst of queries has no throttle mechanism.

**Single-provider dependency.** If the configured LLM or embedding API (OpenAI or Databricks) is down, the entire agent fails to initialize. There is no fallback provider or cached-embedding degradation path.

### Evaluation

**No confidence scoring.** The agent always produces an answer, even when retrieved context is weak or irrelevant. API consumers cannot distinguish high-confidence from low-confidence responses.

**LLM-as-judge without calibration.** The judge model (GPT-4o) scores across 5 dimensions on a 1–5 scale, but has not been calibrated against human raters. Score consistency depends on the judge model version, which may shift with upstream updates.

**Equal-weight scoring dimensions.** All 5 evaluation dimensions (correctness, completeness, faithfulness, relevance, format compliance) are weighted equally. For a safety-critical engineering assistant, faithfulness (no hallucination) should carry more weight than format compliance.

**No failure-mode analysis.** Evaluation only scores successful answers. There are no metrics for false rejections (saying "not available" when the info exists), over-confident answers, or per-category breakdowns (e.g., OTA questions vs. temperature questions).

---

## 4.2 Potential Enhancements

### RAG & Retrieval

**Query rewriting between iterations.** Before each retrieval round, use the LLM to rephrase the query based on what was already retrieved. This makes multi-round retrieval meaningful — each round targets information gaps rather than repeating the same search.

**Adaptive k selection.** Adjust the number of retrieved chunks based on query complexity. Simple factoid questions use k=1–2; cross-series comparisons use k=5+. Can be implemented via a lightweight classifier or LLM metadata on the query.

**Hybrid retrieval (BM25 + vector).** Combine keyword-based retrieval with semantic search. BM25 handles exact technical terms (e.g., "LPDDR4") that embedding models may struggle with, while vector search handles paraphrased queries.

**Persistent vector store.** Replace in-memory Chroma with a persistent store (PostgreSQL + pgvector, Pinecone) to support larger corpora and enable incremental indexing without full rebuilds.

### Agent Framework

**Dynamic iteration limit.** Replace the fixed `MAX_ITERATIONS=3` with a cost-aware or confidence-aware ceiling. The agent could stop early when retrieval similarity scores are high, or extend beyond 3 rounds for genuinely complex queries within a cost budget.

**Tool-level retry with backoff.** Wrap tool execution in a retry mechanism for transient failures (API timeouts, rate limits). Distinguish between retryable errors and permanent failures (unknown tool, malformed query).

**Conversation memory.** Add optional session state persistence so multi-turn workflows don't require full re-specification. A lightweight approach: carry forward the last N messages as context for follow-up questions.

**Input classifier for out-of-scope rejection.** Add a fast, cheap classifier (e.g., a fine-tuned small model or keyword heuristic) before the agent node to hard-reject clearly irrelevant queries without consuming LLM tokens.

### Cost & Operations

**Semantic query cache.** Cache query embeddings and results with a TTL. Identical or near-identical questions (above a cosine similarity threshold) return cached answers, reducing embedding API calls and latency.

**Per-query cost tracking and budget enforcement.** Track token usage per query at runtime. Set a per-query cost ceiling and a daily budget — circuit-break when exceeded to prevent runaway spend.

**Fallback provider chain.** Configure a secondary LLM/embedding provider. If the primary (e.g., OpenAI) fails health checks, route to the fallback (e.g., Databricks-hosted model) automatically.

### Evaluation

**Confidence scoring.** Return a confidence signal alongside each answer — derived from retrieval similarity scores or LLM self-assessment. Allows API consumers to route low-confidence answers to human review.

**Judge calibration.** Run a pilot of 50–100 questions scored by both the LLM judge and human raters. Compute inter-rater agreement (Cohen's kappa) and adjust the judge prompt or scoring rubric to align with human judgment.

**Weighted scoring dimensions.** Assign higher weight to faithfulness and correctness than to format compliance. For a safety-critical domain, a hallucinated specification is far more costly than a formatting preference violation.

**Failure-mode metrics.** Add disaggregated metrics: false rejection rate, per-category accuracy (OTA, temperature, CAN bus, etc.), and over-confidence detection. Enable slicing evaluation results by question type and document source.

> For a detailed scalability analysis with concrete implementation steps, see [SCALABILITY.md](SCALABILITY.md).

---

## 4.3 Success Metrics

The agent is considered production-ready when:

- **Correctness ≥ 4.0 / 5.0** on official 10-question set (release gate)
- **Faithfulness ≥ 4.0 / 5.0** on official set (no hallucination)
- **Latency P99 < 10 seconds** (smoke test)
- **Success rate = 100%** on smoke queries (no API errors)
- **Code coverage ≥ 85%** on unit tests (pylint score > 85)
- **All adversarial queries rejected** (out-of-scope handling)

---

## Summary

The ME Engineering Assistant is a production-grade agentic RAG system built for Bosch's ECU engineering domain, prioritizing correctness, observability, and deployment readiness.

- **Technical Excellence:** Custom LangGraph StateGraph with three query types (single-series, cross-series, out-of-scope), multi-layer anti-hallucination prompts, modular codebase with typed exception hierarchy
- **Production Readiness:** MLflow pyfunc wrapper, quality-gated model registration, Databricks Asset Bundle for one-command deployment, automated smoke test + evaluation pipeline
- **Strategic Thinking:** Documented trade-offs for all key design decisions, evaluation strategy, scalability analysis with concrete migration paths (see [SCALABILITY.md](SCALABILITY.md))
