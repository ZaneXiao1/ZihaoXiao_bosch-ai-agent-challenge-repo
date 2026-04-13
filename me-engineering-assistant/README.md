# ME Engineering Assistant

A production-ready, multi-source RAG agent that answers questions about Electronic Control Unit (ECU) specifications across the ECU-700 and ECU-800 product lines.

---

## Quick Start

```bash
# 1. Install the package (editable mode for development)
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# edit .env and add your OPENAI_API_KEY

# 3. Run the full evaluation suite
python scripts/run_evaluation.py
```

---

## Project Structure

```
me-engineering-assistant/
├── pyproject.toml              # Package definition and dependencies
├── .env.example                # Environment variable template
├── .pylintrc                   # Pylint configuration (score > 85%)
├── evaluation_results.md       # Latest 10/10 evaluation run output
├── README.md
├── src/
│   └── me_assistant/
│       ├── __init__.py
│       ├── config.py           # Model provider abstraction (OpenAI / Databricks)
│       ├── documents/
│       │   ├── __init__.py
│       │   ├── loader.py       # Markdown section-based chunking
│       │   └── store.py        # Dual FAISS index + retriever factory
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── prompts.py      # System prompt with answer discipline rules
│       │   ├── tools.py        # LangChain retrieval tools with fallback
│       │   └── graph.py        # LangGraph ReAct agent (create_react_agent)
│       ├── model/
│       │   ├── __init__.py
│       │   └── mlflow_wrapper.py   # MLflow pyfunc wrapper (log + predict)
│       └── data/               # ECU markdown source documents
│           ├── ECU-700_Series_Manual.md
│           ├── ECU-800_Series_Base.md
│           └── ECU-800_Series_Plus.md
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py       # Vector store and retrieval tests
│   ├── test_agent.py           # Agent integration tests
│   └── test_queries.csv        # 10 predefined test questions + criteria
└── scripts/
    └── run_evaluation.py       # Run all 10 queries and report pass/fail
```

---

## Architectural Design Decisions

### 1. Section-based chunking over fixed-size chunking

Each document is split on its natural section boundaries (`## headings` for ECU-800 files, `**N. Title**` numbered sections for ECU-700 files) rather than by a fixed token count.

**Why:** The ECU documents contain specification tables that must stay intact. A mid-table split would break the structure and cause embedding to miss key values. The documents are small enough (~11 chunks total) that section-level granularity provides both correct boundaries and full context per chunk.

### 2. Context prefix injection into `page_content`

Every chunk's `page_content` is prefixed with its series and model identity before indexing, e.g.:

```
[ECU-700 Series | ECU-750] Diagnostics
The ECU-750 provides basic diagnostic trouble codes (DTCs)…
```

**Why:** Embedding similarity search only sees `page_content`, not the `metadata` dict. Without this prefix, a chunk like "runs a Yocto-based Linux OS" carries zero signal about which ECU it belongs to. The prefix makes every chunk self-contained.

### 3. Two separate FAISS indices (ECU-700 and ECU-800)

Two independent FAISS vector stores are built — one per product series.

**Why:** A single combined index would allow high-similarity ECU-800 chunks (e.g. "OTA Update Capability") to crowd out the low-but-relevant ECU-700 chunk ("OTA updates are not supported") for the query "which models support OTA". Separate indices guarantee both series are always represented when the agent queries for feature availability across the full product line.

### 4. Two-layer multi-source retrieval architecture

- **Layer 1 (LLM reasoning):** The agent LLM reads tool docstrings and autonomously decides which tool(s) to call — single tool for focused queries, both tools for cross-series comparisons. This is pure language understanding, no embedding involved.
- **Layer 2 (Embedding similarity):** Each called tool internally performs cosine similarity search against its dedicated FAISS index to surface the most relevant chunks. This is pure vector math, no LLM involved.

### 5. Full-document fallback inside retrieval tools

When retrieved chunks are empty or fewer than 200 characters, the tool returns the full concatenated document content instead.

**Why:** Documents are < 2 KB each. Passing the full document as context costs ~1 500 tokens — well within any LLM's context window — and guarantees correctness even when the query phrasing has poor semantic overlap with the relevant chunk (e.g. "which supports OTA" vs. "OTA not supported").

### 6. Model provider abstraction via environment variables

All model instantiation is centralised in `config.py`. Switching from OpenAI (local dev) to Databricks (production) requires only changing `MODEL_PROVIDER` — zero code changes.

### 7. `text-embedding-3-small` over `text-embedding-3-large`

The corpus is ~11 short chunks (~6 KB total). The small model (1 536 dimensions) provides more than sufficient discrimination at this scale. The large model (3 072 dimensions) adds latency and cost with no measurable accuracy benefit for a corpus of this size.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PROVIDER` | `openai` | `openai` or `databricks` |
| `OPENAI_API_KEY` | — | Required when `MODEL_PROVIDER=openai` |
| `DATABRICKS_HOST` | — | Required when `MODEL_PROVIDER=databricks` |
| `DATABRICKS_TOKEN` | — | Required when `MODEL_PROVIDER=databricks` |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | LLM model name or Databricks endpoint |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Embedding model name or endpoint |

---

## Running Tests

```bash
# Unit tests only (no API key required)
pytest tests/ -v -m "not integration"

# All tests including integration (requires OPENAI_API_KEY)
pytest tests/ -v

# Pylint quality check
pylint src/me_assistant
```

---

## Limitations & Future Work

- **Corpus size:** Designed for a small, fixed set of three documents. Scaling to hundreds of documents would require persistent vector stores (e.g. Chroma, Pinecone) and incremental indexing.
- **Chunk granularity:** Section-based splitting produces large chunks for spec tables. A hybrid approach (section split → secondary token-aware split) could improve retrieval precision at scale.
- **Evaluation:** The current pass/fail evaluation uses keyword matching. Production validation should use MLflow evaluation with expert-rated golden datasets and semantic similarity metrics.
