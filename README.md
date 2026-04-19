# ME Engineering Assistant

A production-ready Retrieval-Augmented Generation (RAG) agent that answers technical questions about Electronic Control Unit (ECU) specifications across the ECU-700 and ECU-800 product lines. Built with LangGraph, LangChain.

---

## Table of Contents

1. [Architectural Design](#architectural-design)
2. [Setup & Deployment](#setup--deployment)
3. [Testing & Validation Strategy](#testing--validation-strategy)
4. [Limitations & Future Work](#limitations--future-work)

---

# 1. Architectural Design

## 1.1 System Overview

The ME Engineering Assistant is a three-layer RAG architecture:

```
┌──────────────────────────────────────────┐
│  User Query (Natural Language)            │
└──────────────────────────────┬────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  LLM Agent Reasoning   │
                    │  (Tool Selection)      │
                    └──────────┬─────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
    ┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼─────────┐
    │ ECU-700 Index  │ │ ECU-800 Index│ │ Force Answer     │
    │ (Vec DB)       │ │ (Vec DB)     │ │ (No Tools)       │
    └────────────────┘ └──────────────┘ └──────────────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  LLM Answer Gen     │
                    │  (with Citations)   │
                    └─────────────────────┘
```

The system handles three key responsibilities:
1. **LLM reasoning** — deciding which retrieval tools to invoke
2. **Semantic search** — retrieving relevant chunks from vector stores
3. **Answer generation** — producing grounded, cited responses

## 1.2 Agent Graph Architecture

The agent is implemented as a custom LangGraph `StateGraph` with three named nodes and explicit state management. This replaces the prebuilt `create_react_agent` black box with a production-grade, observable graph.

**Key design principle:** Agent autonomously decides when to stop (by not emitting tool_calls), while the graph enforces a deterministic 3-iteration safety ceiling via routing logic.

### Graph Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Question                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ╔═════════════════╗
                    ║  [AGENT NODE]   ║
                    ║ LLM + Tools     ║
                    ║ Decides if need ║
                    ║ tools or answer ║
                    ╚════╤════════════╝
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    No tools          Tools needed    Tools wanted
    → Answer        & iter < 3       & iter >= 3
         │               │               │
         ▼               ▼               ▼
       END         ╔═════════╗      ╔────────────╗
    (return)       ║TOOLS    ║      ║FORCE_      ║
                   ║NODE     ║      ║ANSWER      ║
                   ║Search & ║      ║(no tools)  ║
                   ║increment║      ╚──────┬─────╝
                   ╚────┬────╝             │
                        │                 │
                        └────────┬────────┘
                                 │
                                 ▼
                              (return)
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
┌─────────────────────────────────────────────────────────────┐
│ THE AGENT LOOP                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Agent runs]                                              │
│      ↓                                                      │
│  Agent sees: full conversation + all previous tool results  │
│      ↓                                                      │
│  Agent decides: "Do I have enough info to answer?"         │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ IF NO tool_calls emitted:                           │   │
│  │   → ROUTE to END (return answer now)                │   │
│  │                                                     │   │
│  │ ELSE IF tool_calls emitted AND iteration < 3:      │   │
│  │   → ROUTE to tools node                            │   │
│  │   → tools_node: execute retrieval                  │   │
│  │   → increment iteration_count                      │   │
│  │   → LOOP BACK to agent (agent runs again)          │   │
│  │                                                     │   │
│  │ ELSE IF tool_calls emitted BUT iteration >= 3:     │   │
│  │   → ROUTE to force_answer (NOT tools!)             │   │
│  │   → force_answer: generate answer WITHOUT tools    │   │
│  │   → ROUTE to END                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
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

## 1.4 Exception Hierarchy

The agent defines five custom exception types for clear error classification:

```python
# exceptions.py
MEAssistantError (base)
├── ConfigurationError        # Invalid provider config, missing API keys
├── EmbeddingError            # Vector store creation failed
├── RetrievalError            # Tool invocation or chunk retrieval failed
├── LLMError                  # LLM API auth, rate limit, timeout, or malformed response
└── PreprocessingError        # Document loading or chunking failed
```

Each exception:
- Inherits from a common base for unified error handling
- Includes a descriptive message
- Is caught at the agent boundary and converted to a structured error response

**Why this structure:**
- API consumers (REST endpoint, Databricks jobs) can programmatically distinguish failure modes
- Error messages are prefixed by category (e.g., `"Configuration error: ..."`) for routing
- Operators can set up targeted alerts or retries

---

# 2. Setup & Deployment

## 2.1 Deployment Context

**Note on Databricks Access:** This challenge does not provide Databricks workspace credentials. Following official guidance, this solution implements **manual MLflow logging with detailed deployment plan** as the deployment approach.

**What's Ready:**
- **MLflow pyfunc model** (`src/me_assistant/model/mlflow_wrapper.py`) — implements required `predict()` interface
- **Manual logging script** (`scripts/train_and_log_model.py`) — executes evaluation and logs metrics to MLflow
- **Detailed deployment specification** (Sections 2.2–2.4) — complete DAB configuration and deployment steps

Upon receiving Databricks workspace access, the solution deploys without code changes following the DAB steps in Section 2.2.

---

## 2.2 ⭐ Quick Start (Local Development)

### Prerequisites
- Python 3.10+
- `pip` or `uv` for package management
- OpenAI API key (for local development)

### Installation

```bash
# 1. Clone and install (editable mode for development)
cd me-engineering-assistant
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Verify installation
python -c "from me_assistant.agent.graph import create_agent; print('Installation successful')"
```

### Running the Agent (Local)

Start the interactive chat CLI:

```bash
python scripts/chat_cli.py
```

The script initializes the agent and displays:
```
Initialising ME Engineering Assistant...
Ready. Ask a question about ECU-700 / ECU-800 documentation.
Type 'exit' or 'quit' to stop.
```

Then type your questions at the `You>` prompt and press Enter. The agent responds immediately. Type `exit` or `quit` to end the session.

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
# databricks.yml (simplified)
targets:
  dev:
    workspace:
      host: "https://your-dev-workspace.databricks.com"
  prod:
    workspace:
      host: "https://your-prod-workspace.databricks.com"

resources:
  jobs:
    train_and_log_model:
      tasks:
        - task_key: main
          notebook_path: ./scripts/train_and_log_model
          environment_variables:
            MODEL_PROVIDER: "databricks"
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
- **Detailed deployment plan** (Sections 2.1–2.2) — fully specifies how to deploy to Databricks with DAB configuration

Upon receiving Databricks workspace access, deployment would follow the steps outlined in Section 2.2 without any code changes.

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

### Corpus Size & Scalability
**Current:** The agent is optimized for 3 small documents (~11 chunks total).  
**Limitation:** Scaling to hundreds of documents would require:
- Persistent vector stores (Chroma persistent mode, Pinecone, Weaviate)
- Incremental indexing pipelines (avoid rebuilding all embeddings on update)
- Distributed chunk storage and retrieval

**Why it matters:** In-memory Chroma stores scale to ~10K chunks; beyond that, memory and latency degrade.

### Chunk Granularity
**Current:** Section-based chunking produces variable-size chunks (small preambles to large tables).  
**Limitation:** Section boundaries don't always align with optimal retrieval granularity.

**Future improvement:** Hybrid splitting — split sections on markdown level-3 headings or at ~512 tokens, but preserve table integrity via post-processing.

### Evaluation Cost
**Current:** Full LLM-as-judge eval on 60 questions costs ~$0.05–$0.10.  
**Limitation:** High-frequency CI (e.g., on every commit) becomes expensive.

**Future approach:** Tiered evaluation
- Fast tier: 10-question keyword check (free)
- Full tier: 60-question GPT-4o eval (on-demand, e.g., before release)
- Budget: ~$10/month for daily full evals

### Agent Iteration Limit
**Current:** Hard-coded ceiling of 3 retrieval rounds.  
**Limitation:** Some complex queries might benefit from iterative refinement beyond 3 rounds.

**Future improvement:** Dynamic iteration limit based on query complexity (e.g., estimated via embedding similarity or LLM metadata)

### No Confidence Scoring
**Current:** Agent always produces an answer, even if retrieved context is weak.  
**Limitation:** API consumers cannot distinguish high-confidence from low-confidence answers.

**Future enhancement:** Add a confidence threshold check — if retrieved context similarity falls below a threshold, route the query to human review or return a "confidence too low" signal.

### Limited Context Window Usage
**Current:** Each chunk carries a context prefix, but doesn't use the full LLM context window.  
**Limitation:** On complex multi-part questions, passing more context (e.g., all retrieved chunks + full documents) might improve answers.

**Future optimization:** Adaptive context packing — estimate answer quality from initial retrieval; if low, pack additional context and re-prompt.

---

## 4.2 Potential Enhancements

### Query Expansion & Rewriting
**Idea:** Use an LLM to expand or rephrase user queries before embedding, improving semantic overlap.

**Implementation:**
```python
# Before retrieval
expanded_query = llm.invoke(
    "Rephrase this query to improve semantic search: " + user_question
)
```

**Cost:** One additional LLM call per query (~0.01 tokens).  
**Benefit:** Improved retrieval recall for paraphrased or ambiguous queries.

### Metadata Filtering
**Idea:** Use metadata filters (series, model) to narrow search space before embedding.

**Implementation:**
```python
# Add where clause to Chroma search
results = vector_store.similarity_search(
    query,
    where={"series": "ECU-800"}
)
```

**Benefit:** Faster retrieval, reduced hallucination on cross-series queries.  
**Trade-off:** Requires explicit series mention in user query.

### Hybrid Retrieval (BM25 + Vector)
**Idea:** Combine keyword-based retrieval (BM25) with vector search for robustness.

**Implementation:**
```python
keyword_results = bm25_retriever.invoke(query)
vector_results = vector_retriever.invoke(query)
merged = rerank(keyword_results + vector_results)
```

**Benefit:** Recovers from poor embeddings on technical jargon.  
**Cost:** Added latency (~2–5x slower).

### Fine-Tuned Embeddings
**Idea:** Fine-tune `text-embedding-3-small` on ECU domain examples.

**Implementation:** Train on 100–500 examples of (query, relevant_chunk) pairs.  
**Benefit:** Higher semantic precision for domain-specific terminology.  
**Cost:** Data annotation effort, ongoing model maintenance.

### Structured Output
**Idea:** Return structured JSON instead of plain text.

**Implementation:**
```python
output_schema = {
    "answer": str,
    "sources": list[str],
    "confidence": float,
    "reasoning": str
}
```

**Benefit:** API consumers can parse structure; improve downstream processing.

---

## 4.3 Known Trade-Offs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Indexing | Dual-index (7 + 4 chunks) | Single index (11 chunks) | Prevents ECU-800 dominance on cross-series queries |
| Chunking | Section-based | Fixed-token (512) | Preserves table integrity |
| Embedding | `text-embedding-3-small` | `-large` (3K dims) | No accuracy gain at 11-chunk scale |
| Iteration | Hard limit (3) | Adaptive (confidence-based) | Deterministic safety; prevents infinite loops |
| State | Explicit (graph-based) | Message-only | Enables per-node diagnostics without LLM hacks |

---

## 4.4 Success Metrics

The agent is considered production-ready when:

- **Correctness ≥ 4.0 / 5.0** on official 10-question set (release gate)
- **Faithfulness ≥ 4.0 / 5.0** on official set (no hallucination)
- **Latency P99 < 10 seconds** (smoke test)
- **Success rate = 100%** on smoke queries (no API errors)
- **Code coverage ≥ 85%** on unit tests (pylint score > 85)
- **All adversarial queries rejected** (out-of-scope handling)

---

## Summary

The ME Engineering Assistant is a **production-grade RAG agent** built on:
- **Explicit agent graph** with deterministic safety limits
- **Dual-index retrieval** for balanced coverage across product lines
- **Section-based chunking** optimized for technical specifications
- **Model abstraction** enabling seamless OpenAI ↔ Databricks switching
- **Comprehensive evaluation** with release gates and continuous monitoring

The architecture prioritizes **correctness, observability, and operational simplicity** over generality, making it suitable for Bosch's ECU engineering domain and extensible to related product lines.
