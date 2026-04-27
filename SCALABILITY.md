# Scalability Analysis

Concrete implementation steps for scaling the ME Engineering Assistant from a 3-document prototype to a production system handling large corpora, high traffic, and diverse document formats.

---

## 1. Data Scale: In-Memory Chroma to Milvus

### Problem

The current system uses in-memory Chroma stores (~11 chunks). This works for prototyping but breaks at scale:
- Memory grows linearly with corpus size
- No persistence — full rebuild on every restart
- No horizontal scaling — single-process bottleneck

### Migration Path

**Step 1 — Persistent Chroma (< 10K chunks)**

Replace `Chroma()` with persistent mode as an intermediate step:

```python
vector_stores = {
    "ecu_700": Chroma(
        collection_name="ecu-700",
        embedding_function=embedder,
        persist_directory="./chroma_db/ecu_700",
    ),
}
```

This eliminates rebuild-on-restart with minimal code change. Sufficient for up to ~10K chunks.

**Step 2 — Milvus (10K–10M+ chunks)**

For industrial-scale deployment, migrate to Milvus:

```python
from langchain_milvus import Milvus

vector_stores = {
    "ecu_700": Milvus(
        collection_name="ecu_700",
        embedding_function=embedder,
        connection_args={"host": "milvus-host", "port": "19530"},
    ),
}
```

Why Milvus over alternatives:
- Distributed architecture — scales horizontally across nodes
- GPU-accelerated indexing (IVF_FLAT, HNSW) — handles 100M+ vectors
- Native partition support — replace the dual-index strategy with partitions per ECU series, extensible to N series without code changes
- Built-in incremental indexing — new documents are indexed without rebuilding existing embeddings

**Step 3 — Incremental indexing pipeline**

Decouple document ingestion from agent startup:

```
Document update → Chunking job → Embed → Upsert to Milvus
                                              ↓
                               Agent reads from Milvus at query time
```

The agent no longer rebuilds vector stores on startup. Document updates take effect within seconds via upsert, not minutes via redeployment.

---

## 2. Chunking Strategy: Section-Based to Hierarchical Chunking

### Problem

The current system uses section-based chunking — each markdown section (e.g., `## Specifications`, `**3. Diagnostics**`) becomes one chunk. This works well for 3 small documents (~11 chunks total) where sections are naturally well-scoped. At scale it breaks down:

- **Oversized chunks.** Some sections in industrial manuals span 2000+ tokens (full specification tables, multi-page diagnostic procedures). These exceed the embedding model's effective window — information at the end of a long chunk gets diluted in the embedding vector, reducing retrieval precision.
- **Undersized chunks.** Short sections (e.g., a one-line revision note) produce chunks with too little context to be useful on their own.
- **No overlap.** When relevant information spans a section boundary, neither chunk contains the full context. The LLM receives a truncated view.

### Migration Path

**Step 1 — Hierarchical chunking with token-aware splitting**

Replace the fixed section-based split with a three-level hierarchy that adapts to content length:

```
Level 1: Section (## heading)          → keep as-is if ≤ 512 tokens
Level 2: Paragraph split              → if section > 512 tokens, split on double newlines
Level 3: Sentence split               → if paragraph > 512 tokens, split on sentence boundaries
```

```python
import tiktoken

_ENCODER = tiktoken.encoding_for_model("text-embedding-3-small")
_MAX_CHUNK_TOKENS = 512
_OVERLAP_TOKENS = 50

def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))

def hierarchical_chunk(section_text: str, max_tokens: int = _MAX_CHUNK_TOKENS,
                       overlap_tokens: int = _OVERLAP_TOKENS) -> list[str]:
    """Split a section into chunks using a three-level hierarchy."""
    # Level 1: section fits within limit — return as-is
    if _count_tokens(section_text) <= max_tokens:
        return [section_text]

    # Level 2: split on paragraphs (double newline)
    paragraphs = re.split(r"\n\n+", section_text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        candidate = (current_chunk + "\n\n" + para).strip()
        if _count_tokens(candidate) <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Level 3: paragraph itself too long — split on sentences
            if _count_tokens(para) > max_tokens:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub_chunk = ""
                for sent in sentences:
                    sub_candidate = (sub_chunk + " " + sent).strip()
                    if _count_tokens(sub_candidate) <= max_tokens:
                        sub_chunk = sub_candidate
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Add overlap: prepend the last N tokens of the previous chunk
    if overlap_tokens > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap_tokens)

    return chunks
```

**Step 2 — Context-preserving overlap**

Each chunk gets a small overlap window from the previous chunk, so boundary information is never lost:

```python
def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Prepend the tail of each previous chunk to the next chunk."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _ENCODER.encode(chunks[i - 1])
        overlap_text = _ENCODER.decode(prev_tokens[-overlap_tokens:])
        result.append(f"...{overlap_text}\n\n{chunks[i]}")
    return result
```

Example with a 2000-token specification section:

```
Before (section-based):
  [1 chunk, 2000 tokens] → embedding dilution, tail information lost

After (hierarchical + overlap):
  [Chunk 1: 480 tokens]  ← paragraphs 1-3
  [Chunk 2: 500 tokens]  ← 50-token overlap + paragraphs 4-6
  [Chunk 3: 490 tokens]  ← 50-token overlap + paragraphs 7-9
  [Chunk 4: 320 tokens]  ← 50-token overlap + remaining paragraphs
```

**Step 3 — Parent-child retrieval (optional, advanced)**

For maximum precision, store chunks at two granularities:

```
Parent chunk (full section, ~1000 tokens)  → stored for LLM context
Child chunks (paragraphs, ~200 tokens)     → stored for retrieval
```

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

Retrieval matches on small, precise child chunks (high relevance), but the LLM receives the full parent chunk (complete context). This avoids the trade-off between retrieval precision and generation context.

### Why this matters

| Metric | Section-based (current) | Hierarchical + overlap |
|---|---|---|
| Chunk size consistency | Varies 50–2000 tokens | Controlled ≤ 512 tokens |
| Boundary information loss | Yes | No (overlap) |
| Retrieval precision at scale | Degrades | Stable |
| Works for 3 docs | Yes | Yes |
| Works for 10K+ docs | No | Yes |

---

## 3. Retrieval Strategy: Hybrid Search with BM25 + Semantic Reranking

### Problem

Pure vector search has a known weakness: it can miss exact keyword matches (e.g., part numbers like "ECU-750-A2", specific voltage values "3.3V") because embeddings capture semantic meaning, not lexical precision. Conversely, pure keyword search (BM25) misses paraphrases and synonyms. Neither approach alone is sufficient for industrial-grade retrieval at scale.

### Implementation

Hybrid search combines both to get the best of each:

```
User Query
    │
    ├──→ BM25 (keyword match)     → Top 20 candidates
    │                                      │
    ├──→ Vector Search (semantic)  → Top 20 candidates
    │                                      │
    └──→ Reciprocal Rank Fusion (RRF)      │
              merge + deduplicate ←────────┘
                      │
                      ▼
              Top 20 merged results
                      │
                      ▼
              Cross-Encoder Reranker (e.g., ms-marco-MiniLM)
                      │
                      ▼
              Top 5 final chunks → LLM
```

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. Build BM25 retriever from the same documents
bm25_retriever = BM25Retriever.from_documents(documents, k=20)

# 2. Existing vector retriever
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

# 3. Merge with Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # slightly favor semantic
)

# 4. Rerank with cross-encoder
reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(model=reranker, top_n=5),
    base_retriever=ensemble_retriever,
)
```

### Why each layer matters

- **BM25** catches exact matches that embeddings miss (part numbers, model IDs, specific values)
- **Vector search** catches semantic matches that keywords miss ("power consumption" ↔ "energy usage")
- **RRF fusion** combines both ranked lists without needing score normalization — robust and parameter-free
- **Cross-encoder reranker** scores each (query, chunk) pair with full cross-attention, far more accurate than cosine similarity for final ranking
- At 10K+ chunks, the precision gap between naive vector search and hybrid+rerank widens significantly

---

## 4. Document Lifecycle: Handling Frequent Updates

### Problem

The current system loads documents once at startup from hardcoded file paths. Updating a single spec requires code changes and full redeployment. In an industrial setting, ECU specifications are revised frequently — new models are added, existing specs are amended, and obsolete documents are retired.

### Implementation

**Event-driven ingestion pipeline:**

```
Document source (Git repo / S3 / SharePoint)
        │
        ▼  (webhook or scheduled poll)
┌──────────────────┐
│  Change Detector  │  ← detects added, modified, deleted files
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Chunking +       │  ← re-chunks only changed documents
│  Embedding        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vector Store     │  ← upsert new/modified chunks, delete removed ones
│  (Milvus)         │
└──────────────────┘
```

Key design decisions:

- **Change detection, not full rebuild.** Each document is hashed on ingestion. On update, only documents whose hash changed are re-chunked and re-embedded. This reduces a 10K-document update from minutes to seconds.

- **Stable chunk IDs.** Each chunk uses a deterministic ID such as `doc_id::section_id::chunk_index`. This lets the ingestion job upsert the exact changed chunks instead of creating duplicates every time a document is reprocessed.

- **Upsert for added/modified documents.** New documents are chunked, embedded, and inserted into Milvus. Modified documents follow the same path, but matching chunk IDs are overwritten with the latest vector, text, and metadata.

- **Chunk-level versioning.** Each chunk carries a `doc_version` metadata field tied to the source document's commit hash or timestamp. This enables rollback — if a bad document update degrades answer quality, revert to the previous chunk version without re-embedding the entire corpus.

- **Deletion propagation.** When a document is retired, all its chunks are deleted from the vector store by `doc_id` filter. Without this, stale chunks persist and the agent answers from outdated specs.

- **Zero-downtime updates.** The agent reads from the vector store at query time, not at startup. Upserts are atomic at the chunk level — ongoing queries see either the old or new version of a chunk, never a partial state.

**Update flow summary:**

```
New file      → chunk + embed → upsert chunks into Milvus partition
Modified file → hash changed  → re-chunk + re-embed only that file → upsert changed chunks
Deleted file  → doc missing   → delete all chunks where doc_id matches
```

This keeps the serving path simple: the agent always queries Milvus, while a separate ingestion job keeps the index fresh in the background.

**Tool registry auto-update:**

The current system hardcodes two tools (`search_ecu_700_docs`, `search_ecu_800_docs`). At scale with frequent new product lines, tools should be generated dynamically from the vector store's partition list:

```python
partitions = milvus_client.list_partitions("ecu_docs")
# → ["ecu_700", "ecu_800", "ecu_900", ...]

tools = [create_search_tool(partition) for partition in partitions]
llm_with_tools = llm.bind_tools(tools)
```

Adding a new ECU series becomes a data operation (upload documents to a new partition), not a code change.

---

## 5. Multimodal Document Processing

### Problem

The current loader handles only markdown text files. Industrial ECU documentation includes:
- PDF datasheets with embedded tables and circuit diagrams
- CAD drawings and wiring schematics (images)
- Scanned legacy documents (image-based PDFs)

### Implementation

**PDF table extraction:**

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="ECU-900_Datasheet.pdf",
    strategy="hi_res",           # uses layout detection model
    infer_table_structure=True,  # preserves table structure as HTML
)

# Filter and process by element type
tables = [e for e in elements if e.category == "Table"]
text_blocks = [e for e in elements if e.category == "NarrativeText"]
```

Why `unstructured` over PyPDF: layout-aware parsing preserves table structure. PyPDF flattens tables into line-by-line text, destroying row/column relationships that are critical for specification lookup.

**Image and diagram handling:**

```python
from langchain_core.messages import HumanMessage

# Use vision-capable LLM to generate text descriptions of diagrams
response = llm.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "Describe this ECU wiring diagram in detail."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    ])
])

# Index the generated description as a text chunk
chunks.append(Document(
    page_content=f"[ECU-900 | Wiring Diagram] {response.content}",
    metadata={"source": "ECU-900_Wiring.png", "type": "diagram"},
))
```

This converts visual content into searchable text chunks. The context prefix (`[ECU-900 | Wiring Diagram]`) ensures the embedding model captures both the series and content type.

**OCR for scanned documents:**

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="legacy_scan.pdf",
    strategy="ocr_only",
    ocr_languages=["eng", "deu"],  # English + German for Bosch docs
)
```

---

## 6. Query Expansion and Rewriting

### Problem

Repeated retrieval rounds against the same store return identical results because the same query produces the same embeddings. Multi-round retrieval adds latency without new information.

### Implementation

Add a query rewriting step in `agent_node` when the agent decides to call tools again after seeing previous results:

```python
def agent_node(state: AgentState, llm_with_tools) -> dict:
    messages = list(state["messages"])

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # On iterations > 0, inject a rewriting instruction
    if state["iteration_count"] > 0:
        messages.append(SystemMessage(content=(
            "You are about to search again. Reformulate your search query "
            "to target information NOT already present in the previous tool "
            "results. Use different keywords, ask about related attributes, "
            "or broaden/narrow the scope."
        )))

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

This gives the LLM explicit instruction to vary its query, making subsequent retrieval rounds return different chunks. No changes needed to the graph structure or tool implementations.

**Alternative — embedding-level deduplication:**

```python
def _retrieve_chunks(retriever, query: str, seen_ids: set) -> str:
    docs = retriever.invoke(query)
    new_docs = [d for d in docs if d.metadata.get("chunk_id") not in seen_ids]
    return "\n\n".join(doc.page_content for doc in new_docs)
```

Track seen chunk IDs in `AgentState` and filter them out on subsequent rounds. This guarantees new information per round but requires adding `seen_chunk_ids` to the state schema.

---

## 7. Traffic: Async Processing and Concurrency

### Problem

The current agent runs synchronously — `agent.invoke()` blocks until the full graph completes. Under concurrent traffic, each request holds a thread for 2–10 seconds (LLM latency), limiting throughput to ~10 concurrent users on a single instance.

### Implementation

**Step 1 — Async graph execution**

LangGraph natively supports async invocation:

```python
# Current (blocking)
result = agent.invoke(initial_state)

# Async
result = await agent.ainvoke(initial_state)
```

All node functions need async counterparts:

```python
async def agent_node(state: AgentState, llm_with_tools) -> dict:
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}
```

This allows a single process to handle 50–100 concurrent requests with an async framework (FastAPI, Starlette).

**Step 2 — Request queue with worker pool**

For traffic spikes beyond single-instance capacity:

```
Client → FastAPI → Redis Queue → Worker Pool (N instances)
                                       ↓
                                  Agent.ainvoke()
                                       ↓
                              Redis → Response to Client
```

Each worker runs an independent agent instance with its own LLM connection. The queue absorbs burst traffic and provides backpressure when workers are saturated.

**Step 3 — Streaming responses**

For long-running queries, stream partial results to reduce perceived latency:

```python
async def stream_query(question: str):
    async for event in agent.astream_events(initial_state, version="v2"):
        if event["event"] == "on_chat_model_stream":
            yield event["data"]["chunk"].content
```

The client receives tokens as they are generated rather than waiting for the full answer. Critical for queries that trigger multiple retrieval rounds (6–15 seconds total).

---

## 8. Semantic Query Cache

### Problem

Common questions (e.g., "What is the max operating temperature for ECU-750?") are asked repeatedly. Each instance re-embeds the query, re-searches the vector store, and re-generates the answer — wasting API calls and adding latency.

### Implementation

```python
import hashlib
from redis import Redis

redis = Redis(host="cache-host", port=6379)
CACHE_TTL = 3600  # 1 hour
SIMILARITY_THRESHOLD = 0.95

def query_with_cache(agent, question: str) -> dict:
    # 1. Compute embedding for cache lookup
    query_embedding = embedder.embed_query(question)
    cache_key = hashlib.sha256(question.encode()).hexdigest()

    # 2. Check exact match
    cached = redis.get(f"exact:{cache_key}")
    if cached:
        return json.loads(cached)

    # 3. Check semantic near-match (optional, requires vector-capable cache)
    # If a stored query has cosine similarity > 0.95, return its cached answer

    # 4. Cache miss — run agent
    result = query_agent(agent, question)
    redis.setex(f"exact:{cache_key}", CACHE_TTL, json.dumps(result))
    return result
```

Two cache tiers:
- **Exact match** (hash-based) — zero-cost lookup, catches identical questions
- **Semantic match** (embedding similarity > 0.95) — catches paraphrases like "max temp for ECU-750" vs "ECU-750 maximum operating temperature"

Cache invalidation: TTL-based (1 hour default), plus manual flush when documents are updated via the incremental indexing pipeline.

---

## 9. Summary: Migration Priority

| Enhancement | Effort | Impact | When to Implement |
|---|---|---|---|
| Persistent Chroma | Low | Medium | Immediately — eliminates rebuild-on-restart |
| Hierarchical chunking + overlap | Low | High | When documents have sections > 512 tokens |
| Event-driven ingestion | Medium | High | When documents are updated more than weekly |
| Async execution | Low | High | Before any multi-user deployment |
| Semantic query cache | Medium | High | When query volume exceeds ~100/day |
| Query rewriting | Low | Medium | When multi-round retrieval is observed to repeat |
| Hybrid search (BM25 + rerank) | Medium | High | When retrieval precision matters (10K+ chunks) |
| Milvus migration | High | High | When corpus exceeds ~10K chunks |
| Dynamic tool registry | Medium | High | When new ECU series are added regularly |
| Multimodal processing | High | High | When PDF/image documents are added to corpus |
| Streaming responses | Medium | Medium | When P95 latency exceeds user tolerance (~5s) |
| Request queue + workers | Medium | High | When concurrent users exceed ~50 |
