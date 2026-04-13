"""
Retrieval quality evaluation for the RAG system.

For each test question, performs similarity search on the relevant vector
stores and records every returned chunk with its rank, similarity score,
source file, section title, and a content preview. This allows manual
inspection of whether the most relevant chunks are ranked highest.

Does NOT require calling the full agent — queries the vector stores directly
so results are independent of the LLM's tool-calling decisions.
"""
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from me_assistant.config import get_embeddings
from me_assistant.documents.store import build_vector_stores

logger = logging.getLogger(__name__)

DEFAULT_TEST_SET = Path(__file__).parent.parent.parent.parent / "tests" / "test_queries.csv"

# How many chunks to retrieve per store (should match agent's k)
DEFAULT_K = 3


@dataclass
class ChunkResult:
    """One retrieved chunk with its rank and metadata."""
    rank: int
    score: float
    source: str
    section: str
    models_covered: list[str]
    content_preview: str  # first N characters of the chunk


@dataclass
class RetrievalResult:
    """Retrieval results for one test question."""
    question_id: int
    category: str
    question: str
    expected_answer: str
    store_queried: str  # "ecu_700", "ecu_800", or "both"
    chunks: list[ChunkResult]

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "store_queried": self.store_queried,
            "num_chunks": len(self.chunks),
            "chunks": [
                {
                    "rank": c.rank,
                    "score": round(c.score, 4),
                    "source": c.source,
                    "section": c.section,
                    "models_covered": c.models_covered,
                    "content_preview": c.content_preview,
                }
                for c in self.chunks
            ],
        }


def load_test_cases(csv_path: Optional[str] = None) -> list[dict]:
    """Load test cases from CSV."""
    path = Path(csv_path) if csv_path else DEFAULT_TEST_SET
    cases = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "question_id": int(row["Question_ID"]),
                "category": row["Category"],
                "question": row["Question"],
                "expected_answer": row["Expected_Answer"],
                "evaluation_criteria": row["Evaluation_Criteria"],
            })
    return cases


def _determine_stores_to_query(category: str, question: str) -> list[str]:
    """Decide which vector stores to query based on question category/content.

    Mirrors the logic the agent would use:
    - Questions mentioning both series or "compare" / "across" → both stores
    - Questions about ECU-700/750 only → ecu_700
    - Questions about ECU-800/850/850b only → ecu_800
    - Ambiguous → both stores
    """
    q_lower = question.lower()
    cat_lower = category.lower()

    is_comparative = any(kw in cat_lower for kw in ("comparative", "cross"))
    mentions_both = ("750" in q_lower and ("850" in q_lower or "800" in q_lower))
    has_compare_keyword = any(kw in q_lower for kw in ("compare", "across all", "which ecu"))

    if is_comparative or mentions_both or has_compare_keyword:
        return ["ecu_700", "ecu_800"]

    # "all ecu models" / "all models" → both
    if "all" in q_lower and "model" in q_lower:
        return ["ecu_700", "ecu_800"]

    if "700" in q_lower or "750" in q_lower:
        return ["ecu_700"]
    if "800" in q_lower or "850" in q_lower:
        return ["ecu_800"]

    # Default: query both
    return ["ecu_700", "ecu_800"]


def _query_store_with_scores(store, query: str, k: int) -> list[tuple]:
    """Query a Chroma store and return (Document, score) pairs sorted by relevance.

    Uses similarity_search_with_score which returns (doc, distance).
    Lower distance = more similar for Chroma's default L2 metric.
    We convert to a similarity score where higher = better.
    """
    results = store.similarity_search_with_score(query, k=k)
    # Chroma returns (doc, distance); convert distance to similarity
    # similarity = 1 / (1 + distance) so it's in [0, 1], higher is better
    scored = []
    for doc, distance in results:
        similarity = 1.0 / (1.0 + distance)
        scored.append((doc, similarity))
    # Already sorted by distance (ascending) from Chroma, which means
    # highest similarity first after conversion
    return scored


def evaluate_retrieval_single(
    stores: dict,
    test_case: dict,
    k: int = DEFAULT_K,
    preview_length: int = 200,
) -> RetrievalResult:
    """Evaluate retrieval for a single test question.

    Args:
        stores: Dict from build_vector_stores().
        test_case: Test case dict.
        k: Number of chunks to retrieve per store.
        preview_length: Number of chars for content preview.

    Returns:
        RetrievalResult with ranked chunk details.
    """
    question = test_case["question"]
    store_keys = _determine_stores_to_query(test_case["category"], question)

    all_chunks = []
    for store_key in store_keys:
        store = stores[store_key]
        results = _query_store_with_scores(store, question, k=k)
        for doc, score in results:
            all_chunks.append((doc, score))

    # Sort all chunks by score descending (most relevant first)
    all_chunks.sort(key=lambda x: x[1], reverse=True)

    chunk_results = []
    for rank, (doc, score) in enumerate(all_chunks, 1):
        preview = doc.page_content[:preview_length].replace("\n", " ")
        chunk_results.append(ChunkResult(
            rank=rank,
            score=score,
            source=doc.metadata.get("source", "unknown"),
            section=doc.metadata.get("section", "unknown"),
            models_covered=doc.metadata.get("models_covered", []),
            content_preview=preview,
        ))

    store_label = " + ".join(store_keys) if len(store_keys) > 1 else store_keys[0]

    return RetrievalResult(
        question_id=test_case["question_id"],
        category=test_case["category"],
        question=question,
        expected_answer=test_case["expected_answer"],
        store_queried=store_label,
        chunks=chunk_results,
    )


def run_retrieval_evaluation(
    csv_path: Optional[str] = None,
    k: int = DEFAULT_K,
) -> list[RetrievalResult]:
    """Run retrieval evaluation for all test cases.

    Args:
        csv_path: Path to test CSV. Defaults to tests/test_queries.csv.
        k: Number of chunks to retrieve per store.

    Returns:
        List of RetrievalResult, one per test case.
    """
    stores = build_vector_stores()
    test_cases = load_test_cases(csv_path)
    results = []

    logger.info("Running retrieval evaluation on %d questions (k=%d)...", len(test_cases), k)
    for tc in test_cases:
        result = evaluate_retrieval_single(stores, tc, k=k)
        results.append(result)
        logger.info(
            "Q%d [%s]: queried %s → %d chunks",
            tc["question_id"], tc["category"], result.store_queried, len(result.chunks),
        )

    return results


def print_retrieval_report(results: list[RetrievalResult]):
    """Print a human-readable retrieval report for manual review."""
    print("\n" + "=" * 100)
    print("  RAG Retrieval Quality Report — Chunk Ranking Inspection")
    print("=" * 100)

    for r in results:
        print(f"\n{'─' * 100}")
        print(f"  Q{r.question_id} [{r.category}]")
        print(f"  Question:  {r.question}")
        print(f"  Expected:  {r.expected_answer[:120]}...")
        print(f"  Stores:    {r.store_queried} | Chunks returned: {len(r.chunks)}")
        print(f"  {'─' * 96}")
        print(f"  {'Rank':>4}  {'Score':>7}  {'Source':<30}  {'Section':<25}  Content Preview")
        print(f"  {'─' * 96}")

        for c in r.chunks:
            preview = c.content_preview[:60].replace("\n", " ")
            print(f"  {c.rank:>4}  {c.score:>7.4f}  {c.source:<30}  "
                  f"{c.section:<25}  {preview}...")

    # Summary stats
    print(f"\n{'=' * 100}")
    print("  SUMMARY")
    print(f"{'─' * 100}")
    total_chunks = sum(len(r.chunks) for r in results)
    avg_chunks = total_chunks / len(results) if results else 0
    print(f"  Total questions: {len(results)}")
    print(f"  Total chunks retrieved: {total_chunks}")
    print(f"  Avg chunks per question: {avg_chunks:.1f}")

    # Score distribution
    all_scores = [c.score for r in results for c in r.chunks]
    if all_scores:
        print(f"  Score range: {min(all_scores):.4f} — {max(all_scores):.4f}")
        print(f"  Score avg:   {sum(all_scores) / len(all_scores):.4f}")
    print("=" * 100 + "\n")
