# ME Engineering Assistant — Tier 1 Build Specification

## Context

You are building an AI agent for ME Corporation that answers questions about Electronic Control Unit (ECU) specifications. The agent must retrieve information from three markdown documents covering three ECU models (ECU-750, ECU-850, ECU-850b) and provide accurate, synthesized answers.

This is a coding challenge for a Senior AI Engineer position. Code quality, modularity, and architectural decisions matter as much as functionality.

## Success Criteria

- Agent correctly answers **8 out of 10** predefined test queries (see Test Queries section below)
- Response time **< 10 seconds** per query
- Code passes **pylint score > 85%**
- Solution is a **proper installable Python package** (not a notebook or monolithic script)

---

## Step 1: Project Scaffolding

Create the following project structure:

```
me-engineering-assistant/
├── pyproject.toml
├── README.md
├── .pylintrc
├── src/
│   └── me_assistant/
│       ├── __init__.py
│       ├── config.py          # All configuration & model provider abstraction
│       ├── documents/
│       │   ├── __init__.py
│       │   ├── loader.py      # Document loading and chunking
│       │   └── store.py       # Vector store creation and retrieval
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── tools.py       # LangChain retrieval tools
│       │   ├── graph.py       # LangGraph StateGraph definition
│       │   └── prompts.py     # All prompt templates (system, routing, generation)
│       ├── model/
│       │   ├── __init__.py
│       │   └── mlflow_wrapper.py  # MLflow pyfunc model wrapper
│       └── data/
│           ├── ECU-700_Series_Manual.md
│           ├── ECU-800_Series_Base.md
│           └── ECU-800_Series_Plus.md
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_agent.py
│   └── test_queries.csv
└── scripts/
    └── run_evaluation.py      # Run all 10 test queries and report results
```

### pyproject.toml Requirements

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "me-engineering-assistant"
version = "0.1.0"
description = "Multi-source RAG agent for ECU engineering documentation"
requires-python = ">=3.10"
dependencies = [
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "langgraph>=0.2.0",
    "faiss-cpu>=1.7.4",
    "mlflow>=2.15.0",
    "pydantic>=2.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pylint>=3.0",
]
databricks = [
    "langchain-databricks>=0.1.0",
    "databricks-sdk>=0.30.0",
]
```

### config.py — Model Provider Abstraction (CRITICAL DESIGN DECISION)

This is the **most important architectural decision** in the project. The agent must work with OpenAI locally and switch to Databricks models in production by changing only environment variables.

```python
"""
Configuration module with model provider abstraction.

Design Decision: All model instantiation is centralized here.
Switching between OpenAI (local dev) and Databricks (production)
requires only changing environment variables, zero code changes.

Environment Variables:
    MODEL_PROVIDER: "openai" or "databricks"
    OPENAI_API_KEY: Required when MODEL_PROVIDER=openai
    DATABRICKS_HOST: Required when MODEL_PROVIDER=databricks
    DATABRICKS_TOKEN: Required when MODEL_PROVIDER=databricks
    LLM_MODEL_NAME: Model name/endpoint (default varies by provider)
    EMBEDDING_MODEL_NAME: Embedding model name/endpoint
"""
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")


def get_llm():
    """Return the LLM instance based on configured provider."""
    if MODEL_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
        )
    elif MODEL_PROVIDER == "databricks":
        from langchain_databricks import ChatDatabricks
        return ChatDatabricks(
            endpoint=os.getenv("LLM_MODEL_NAME"),
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")


def get_embeddings():
    """Return the embeddings model based on configured provider."""
    if MODEL_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        )
    elif MODEL_PROVIDER == "databricks":
        from langchain_databricks import DatabricksEmbeddings
        return DatabricksEmbeddings(
            endpoint=os.getenv("EMBEDDING_MODEL_NAME"),
        )
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")
```

Create a `.env.example` file:

```
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

**Why `text-embedding-3-small` and not `text-embedding-3-large`?** The corpus is ~11 chunks of short technical text. The small model (1536 dimensions) provides more than sufficient discrimination for this scale. The large model (3072 dimensions) adds latency and cost with no measurable benefit at this corpus size. This is a deliberate tradeoff to document in the README.

---

## Step 2: Document Processing

### loader.py — Chunking Strategy

**Design Decision:** These documents are small (each < 2KB). The chunking strategy must preserve the integrity of specification tables and keep metadata about which ECU series each chunk belongs to. This metadata is critical for the agent's routing logic.

Implementation requirements:

1. **Split by markdown sections** (not by fixed token count). Each `##` heading starts a new chunk. This ensures tables and specification blocks stay intact. Use `MarkdownHeaderTextSplitter` from langchain or implement a simple regex split on `\n## ` or `\n**` top-level section markers.

2. **Attach metadata to every chunk:**
   - `source`: filename (e.g., "ECU-700_Series_Manual.md")
   - `series`: "ECU-700" or "ECU-800" (derived from filename)
   - `models_covered`: list of specific models mentioned (e.g., ["ECU-750"] or ["ECU-850", "ECU-850b"])

3. **Inject a context prefix into every chunk's page_content** so each chunk is self-contained even without metadata. This is critical because embedding search only sees page_content, not metadata. Concrete examples:

   - A chunk from ECU-700_Series_Manual.md about diagnostics should become:
     `"[ECU-700 Series | ECU-750] Diagnostics\nThe ECU-750 provides basic diagnostic trouble codes (DTCs) via the CAN bus interface..."`
   
   - A chunk from ECU-800_Series_Base.md about software should become:
     `"[ECU-800 Series | ECU-850] Software Configuration\nThe ECU-850 runs a Yocto-based Linux OS..."`
   
   - A chunk from ECU-800_Series_Plus.md about NPU should become:
     `"[ECU-800 Series | ECU-850b] NPU Configuration\nTo enable the NPU, use the following driver command..."`

   Without this prefix, a chunk like "runs a Yocto-based Linux OS" has zero indication it belongs to ECU-850, and the embedding will not associate it correctly with queries mentioning "ECU-850".

4. Use LangChain's `Document` class with the metadata dict.

5. **Expected chunk count:** ECU-700 manual → ~4 chunks (Introduction, Technical Specifications, Diagnostics, Safety). ECU-800 Base → ~3 chunks (Overview/Key Features, Technical Specifications, Software Configuration). ECU-800 Plus → ~4 chunks (Overview, Key Differentiators, Full Technical Specifications, NPU Configuration). Total: ~11 chunks. This is intentionally small.

### Fallback Strategy (IMPORTANT — implement this inside the retrieval tools, not as a separate path)

Since documents are very small (each < 2KB), implement a **full-document-as-context fallback** inside each retrieval tool function. The logic:

```python
def _retrieve_with_fallback(retriever, full_doc_content: str, query: str, k: int = 4) -> str:
    """Retrieve relevant chunks, falling back to full document if retrieval quality is low."""
    docs = retriever.invoke(query)
    
    if not docs:
        # No results — return entire document
        return full_doc_content
    
    # Combine retrieved chunks
    retrieved_text = "\n\n".join(doc.page_content for doc in docs)
    
    # If retrieved text is very short (< 200 chars), supplement with full doc
    # This handles cases where the relevant info is in a section that didn't
    # match well semantically (e.g., "OTA not supported" vs "which supports OTA")
    if len(retrieved_text) < 200:
        return full_doc_content
    
    return retrieved_text
```

This is not a separate code path — it's built into the tool itself. The challenge spec explicitly endorses this approach. For ~6KB total corpus, passing a full document as context costs ~1500 tokens, well within any LLM's context window.

---

## Step 3: Vector Store Build

### store.py — FAISS Index Construction

**Design Decision:** Create **two separate FAISS indices** — one for ECU-700 series documents and one for ECU-800 series documents (both ECU-850 and ECU-850b go into the ECU-800 index). This enables the agent to selectively query one or both based on the question type.

**Why not one combined index?** Consider Q7: "Which ECU models support OTA updates?" A combined index would rank ECU-800 chunks about OTA support very high (strong semantic match), potentially pushing ECU-700's "OTA updates are not supported" out of the top-k results — because "not supported" has lower cosine similarity to "which supports OTA" than "OTA Update Capability" does. With separate indices, each tool queries its own index independently, guaranteeing that both series are represented in the agent's context.

**How multi-source retrieval actually works — two layers of intelligence:**
1. **Layer 1 (LLM reasoning, no embedding):** The agent LLM reads the tool docstrings and decides WHICH tool(s) to call based on the question. "ECU-750 temperature?" → call only ecu_700 tool. "Compare ECU-750 and ECU-850 CAN bus?" → call both tools. This routing decision is pure language understanding.
2. **Layer 2 (Embedding similarity, no LLM):** Inside each called tool, the query is vectorized and compared against that tool's FAISS index to find the most relevant chunks. This is pure math.

Implementation requirements:

1. A function `build_vector_stores()` that:
   - Loads and chunks all documents using loader.py
   - Splits chunks by series metadata into two groups
   - Creates two FAISS indices using the embeddings from config.py
   - Stores the full raw content of each document for fallback use
   - Returns a dict: `{"ecu_700": faiss_store_700, "ecu_800": faiss_store_800, "raw_docs": {...}}`

2. A function `get_retriever(series: str, k: int = 4)` that:
   - Returns a LangChain retriever for the specified series
   - Uses similarity search with **k=4** (not 3). Rationale: ECU-700 index has only ~4 chunks, so k=4 essentially returns everything — good, because we never want to miss info from a tiny corpus. ECU-800 index has ~7 chunks, k=4 returns about half, which is sufficient for any single question.
   - Do NOT use a similarity score threshold — with so few chunks, even low-scoring chunks may contain the answer (e.g., "OTA not supported" matching poorly against "which supports OTA")

3. A function `get_all_retrievers(k: int = 4)` that:
   - Returns retrievers for both series (for cross-series comparison queries)

**Important:** The vector stores should be built once and cached in memory (module-level singleton or passed via dependency injection). Do not rebuild them on every query.

---

## Step 4: Retrieval Tools

### tools.py — LangChain Tool Definitions

Create LangChain tools that the agent can call. Each tool wraps a retriever.

```python
"""
Retrieval tools for the LangGraph agent.

Design Decision: Separate tools per document series allows the agent
to intelligently decide which documentation to consult. For cross-series
comparisons, the agent calls both tools.

CRITICAL: The tool docstrings are what the LLM reads to decide which tool
to call. They must clearly describe what each series covers and when to use
each tool. Poor docstrings = wrong routing = wrong answers.
"""
from langchain_core.tools import tool


@tool
def search_ecu_700_docs(query: str) -> str:
    """Search the ECU-700 Series documentation (legacy product line, covers ECU-750).
    
    Use this tool when the question involves ANY of the following:
    - The ECU-700 series or ECU-750 model specifically
    - Legacy/older ECU product specifications
    - Comparing legacy vs newer ECU models (use BOTH this tool and search_ecu_800_docs)
    - Questions about which models support/don't support a feature (must check ALL series)
    
    The ECU-750 is a 32-bit Cortex-M4 based controller for core automotive functions.
    It does NOT support OTA updates. It is certified for ISO 26262 ASIL-B.
    
    Args:
        query: The search query about ECU-700 series specifications.
    
    Returns:
        Relevant documentation excerpts from ECU-700 series manuals.
    """
    # Implementation: call retriever with fallback, format results
    pass


@tool
def search_ecu_800_docs(query: str) -> str:
    """Search the ECU-800 Series documentation (next-gen, covers ECU-850 and ECU-850b).
    
    Use this tool when the question involves ANY of the following:
    - The ECU-800 series, ECU-850 model, or ECU-850b model
    - AI/ML capabilities, NPU, neural processing
    - Next-generation ECU features (OTA updates, secure boot, HSM)
    - Comparing ECU-850 vs ECU-850b (both are in this documentation)
    - Comparing newer vs legacy ECU models (use BOTH this tool and search_ecu_700_docs)
    - Questions about which models support/don't support a feature (must check ALL series)
    
    The ECU-850 is a dual-core Cortex-A53 for ADAS/infotainment.
    The ECU-850b is an AI-enhanced variant with a dedicated NPU.
    
    Args:
        query: The search query about ECU-800 series specifications.
    
    Returns:
        Relevant documentation excerpts from ECU-800 series manuals.
    """
    # Implementation: call retriever with fallback, format results
    pass
```

**Key Implementation Detail for both tools:** Inside each tool function, use the `_retrieve_with_fallback()` pattern from loader.py. The tool should:
1. Call the retriever for its series (k=4)
2. If results are empty or very short, return the full raw document content instead
3. Format the output with clear source attribution, e.g.: `"[Source: ECU-700_Series_Manual.md]\n{content}"`

---

## Step 5: LangGraph Agent

### graph.py — StateGraph Definition

**Design Decision:** Use LangGraph's prebuilt `create_react_agent` or build a custom `StateGraph` with the ReAct pattern. The agent receives a question, decides which tool(s) to call, retrieves context, and generates a final answer.

The agent graph should follow this flow:

```
User Query → LLM (with tools) → [Tool Call Decision]
    ├── search_ecu_700_docs → results
    ├── search_ecu_800_docs → results
    └── (both for comparisons)
→ LLM generates final answer using retrieved context
```

Implementation requirements:

1. **State schema:**
```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

2. **System prompt** (in prompts.py):
```
You are an expert engineering assistant for ME Corporation.
You help engineers find and compare specifications across ECU product lines.

You have access to documentation for two product series:
- ECU-700 Series (legacy): covers the ECU-750 model
- ECU-800 Series (next-gen): covers the ECU-850 and ECU-850b models

Rules:
1. ALWAYS use the search tools to retrieve information before answering.
   Never answer from memory alone.
2. For questions about a SINGLE series, search only the relevant documentation.
3. For COMPARISON questions across series, search BOTH documentation sets.
4. When comparing models, present information in a structured way highlighting
   key differences and similarities.
5. Always cite which specific ECU model(s) your information comes from.
6. If the documentation does not contain the answer, say so explicitly.
   Do not guess or fabricate specifications.
7. Include specific numbers, values, and units in your answers.
```

3. **Graph construction:**
   - Use `create_react_agent` from langgraph with the LLM, tools list, and system prompt
   - OR build a custom StateGraph with: agent_node (LLM call with tool binding) → tool_node (executes tool calls) → loop back or end
   - The agent must handle multi-tool calls (calling both ECU-700 and ECU-800 tools in a single turn for comparison queries)

4. **Entry point function:**
```python
def create_agent():
    """Create and return the compiled LangGraph agent."""
    # Build vector stores
    # Create tools with retriever references  
    # Build graph
    # Compile and return
    pass

def query_agent(agent, question: str) -> str:
    """Run a single query through the agent and return the answer string."""
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content
```

---

## Step 6: MLflow Pyfunc Wrapper

### mlflow_wrapper.py

**Design Decision:** Package the entire agent (vector stores + graph + models) as an MLflow pyfunc model. This is required for Tier 1 ("Basic MLflow model logging with predict() method").

```python
"""
MLflow pyfunc wrapper for the ME Engineering Assistant agent.

This module packages the LangGraph agent as an MLflow model that can be:
1. Logged to MLflow tracking server
2. Loaded and served via MLflow model serving
3. Called via REST API when deployed on Databricks
"""
import mlflow
import pandas as pd
from me_assistant.agent.graph import create_agent, query_agent


class MEAssistantModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the ECU Engineering Assistant."""

    def load_context(self, context):
        """Initialize the agent when the model is loaded.
        
        This is called once when the model is loaded, not on every predict().
        All heavy initialization (vector stores, model loading) happens here.
        """
        self.agent = create_agent()

    def predict(self, context, model_input, params=None):
        """Process user queries and return agent responses.
        
        Args:
            context: MLflow context (unused but required by interface)
            model_input: DataFrame with a 'question' column,
                        or a dict with a 'question' key
            params: Optional parameters (unused)
        
        Returns:
            List of answer strings, one per input question.
        """
        if isinstance(model_input, pd.DataFrame):
            questions = model_input["question"].tolist()
        elif isinstance(model_input, dict):
            questions = [model_input["question"]]
        else:
            questions = [str(model_input)]

        answers = []
        for question in questions:
            try:
                answer = query_agent(self.agent, question)
                answers.append(answer)
            except Exception as e:
                answers.append(f"Error processing query: {str(e)}")
        
        return answers


def log_model():
    """Log the MEAssistantModel to MLflow."""
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        mlflow.pyfunc.log_model(
            artifact_path="me-assistant-model",
            python_model=MEAssistantModel(),
            pip_requirements=[
                "langchain>=0.3.0",
                "langchain-openai>=0.2.0",
                "langchain-community>=0.3.0",
                "langgraph>=0.2.0",
                "faiss-cpu>=1.7.4",
                "mlflow>=2.15.0",
            ],
            # Include document data files as artifacts
            artifacts={
                "ecu_700_doc": "src/me_assistant/data/ECU-700_Series_Manual.md",
                "ecu_800_base_doc": "src/me_assistant/data/ECU-800_Series_Base.md",
                "ecu_800_plus_doc": "src/me_assistant/data/ECU-800_Series_Plus.md",
            },
        )
        print(f"Model logged with run_id: {run.info.run_id}")
        return run.info.run_id
```

---

## Step 7: Validation

### test_queries.csv — The 10 Test Questions

These are the exact queries the agent will be evaluated against:

| ID | Category | Question | Expected Key Points |
|----|----------|----------|-------------------|
| 1 | Single Source - ECU-700 | What is the maximum operating temperature for the ECU-750? | +85°C, range -40°C to +85°C |
| 2 | Single Source - ECU-800 | How much RAM does the ECU-850 have? | 2 GB LPDDR4 |
| 3 | Single Source - ECU-800 Enhanced | What are the AI capabilities of the ECU-850b? | NPU, 5 TOPS |
| 4 | Comparative - Same Series | What are the differences between ECU-850 and ECU-850b? | RAM (2GB vs 4GB), Clock (1.2 vs 1.5 GHz), NPU (5 TOPS) |
| 5 | Comparative - Cross Series | Compare the CAN bus capabilities of ECU-750 and ECU-850. | Single vs Dual channel, 1 Mbps vs 2 Mbps |
| 6 | Technical Specification | What is the power consumption of the ECU-850b under load? | 1.7A under load, 550mA idle |
| 7 | Feature Availability | Which ECU models support Over-the-Air (OTA) updates? | ECU-850 and ECU-850b yes, ECU-750 no |
| 8 | Storage Comparison | How does the storage capacity compare across all ECU models? | 2MB vs 16GB vs 32GB |
| 9 | Operating Environment | Which ECU can operate in the harshest temperature conditions? | ECU-850/850b at +105°C vs ECU-750 at +85°C |
| 10 | Configuration/Usage | How do you enable the NPU on the ECU-850b? | me-driver-ctl --enable-npu --mode=performance |

### run_evaluation.py

Create a script that:
1. Creates the agent
2. Runs all 10 test queries
3. Prints each question, the agent's answer, and the expected key points
4. Measures and reports response time per query
5. Reports overall pass/fail (answer must contain the expected key values)

### tests/test_agent.py

Write pytest tests that:
1. Test that the agent can be created without errors
2. Test a simple single-source query returns relevant content
3. Test a cross-series comparison query calls both retrieval tools
4. Test that response time is under 10 seconds

### pylint

Configure `.pylintrc` with reasonable settings. Target: score > 85%. Key settings:
- max-line-length = 120
- disable=C0114 (missing-module-docstring can be relaxed for __init__.py)

---

## Implementation Order

Execute these steps in this exact order. **Do not skip ahead.**

1. Create the full project structure with all empty files and pyproject.toml
2. Implement config.py with the provider abstraction
3. Implement loader.py with markdown section-based chunking
4. Implement store.py with dual FAISS index creation
5. Implement tools.py with the two retrieval tools
6. Implement prompts.py with the system prompt
7. Implement graph.py with the LangGraph agent
8. Write a quick smoke test — run one query end-to-end to verify it works
9. Implement mlflow_wrapper.py
10. Implement run_evaluation.py and run all 10 test queries
11. Fix any failures (tune prompts, adjust chunking, fix retrieval)
12. Run pylint and fix issues to reach > 85%
13. Write pytest tests

---

## Key Design Decisions to Document in README

When writing the README, explain these decisions and why they were made:

1. **Why section-based chunking over fixed-size?** Tables and spec blocks must stay intact; documents are small enough that section-based chunking preserves context without exceeding token limits.

2. **Why two separate vector stores?** Enables intelligent per-series retrieval and reduces noise. A single store would let high-similarity chunks from one series crowd out relevant chunks from the other — e.g., "OTA Update Capability" (ECU-800) would rank higher than "OTA not supported" (ECU-700) for the query "which models support OTA", causing the agent to miss the negative confirmation. Separate stores guarantee both series are always represented when queried.

3. **How is "multi-source RAG" demonstrated?** Through a two-layer architecture: (1) the LLM agent reads tool docstrings and decides which document source(s) to query — this is reasoning, not embedding; (2) each tool internally performs embedding-based similarity search against its own FAISS index. The agent autonomously calls one tool for single-series questions and both tools for comparison/feature-availability questions. This routing behavior is observable in the LangGraph execution trace.

4. **Why model provider abstraction?** Enables local development with OpenAI while deploying with Databricks models — zero code changes, only environment variable swap.

5. **Why LangGraph ReAct pattern?** The agent needs to autonomously decide which documentation to consult based on the question. Tool-calling with ReAct allows multi-tool invocation for comparison queries.

6. **Why include full-document fallback?** Documents are < 2KB each. If vector retrieval misses a relevant chunk, passing the full document as context is a reliable fallback that ensures accuracy.

---

## Anti-Patterns to Avoid

- **DO NOT** put everything in one file. This is explicitly penalized ("should not be a monolithic notebook").
- **DO NOT** hardcode model names or API keys. Use environment variables.
- **DO NOT** skip type hints and docstrings. They are evaluated under code quality.
- **DO NOT** use print() for debugging in production code. Use Python logging.
- **DO NOT** rebuild vector stores on every query. Build once, reuse.
- **DO NOT** ignore error handling. Wrap tool calls and LLM calls in try/except.
- **DO NOT** use LangChain's deprecated APIs. Use `langchain_core`, `langchain_openai`, `langchain_community` (the split package structure).
