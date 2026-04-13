"""
Vector store creation and retrieval for ECU documentation.

Design Decision: Two separate in-memory vector stores (ECU-700 and ECU-800)
ensure that both series are always represented in the agent's context.
A single combined store risks crowding out low-similarity-but-relevant chunks —
e.g. "OTA updates are not supported" would rank lower than "OTA Update
Capability" for the query "which models support OTA", causing the agent
to miss the negative confirmation from the ECU-700 documentation.

Note: Chroma (in-memory) is used instead of FAISS due to known PyPI wheel
incompatibilities with numpy 1.x on macOS ARM. Chroma is pure-Python and
works identically for this ~12-chunk corpus.
"""
import logging
import uuid
from typing import Optional

from langchain_chroma import Chroma

from me_assistant.config import get_embeddings
from me_assistant.documents.loader import load_documents
from me_assistant.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    EmbeddingError,
    MEAssistantError,
)

logger = logging.getLogger(__name__)

# Module-level cache — avoids rebuilding stores on every query.
_STORE_CACHE: dict = {}


def build_vector_stores(data_dir: Optional[str] = None) -> dict:
    """Build and cache in-memory Chroma vector stores for ECU-700 and ECU-800 series.

    Called once at agent startup; subsequent calls return the cached result.

    Args:
        data_dir: Optional path override for the document directory.

    Returns:
        Dict with keys:
        - ``"ecu_700"``: :class:`~langchain_chroma.Chroma` store
        - ``"ecu_800"``: :class:`~langchain_chroma.Chroma` store
        - ``"raw_docs"``: ``{"ecu_700": str, "ecu_800": str}`` for fallback

    Raises:
        DocumentLoadError: If markdown files cannot be found or read.
        ConfigurationError: If the embedding model cannot be initialised.
        EmbeddingError: If the embedding API fails during indexing.
    """
    if "stores" in _STORE_CACHE:
        return _STORE_CACHE["stores"]

    logger.info("Building vector stores...")

    try:
        documents = load_documents(data_dir)
    except FileNotFoundError as exc:
        raise DocumentLoadError(str(exc)) from exc
    except Exception as exc:
        raise DocumentLoadError(
            f"Failed to load ECU documents: {exc}"
        ) from exc

    try:
        embeddings = get_embeddings()
    except MEAssistantError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to initialise embedding model: {exc}"
        ) from exc

    docs_700 = [d for d in documents if d.metadata["series"] == "ECU-700"]
    docs_800 = [d for d in documents if d.metadata["series"] == "ECU-800"]

    if not docs_700:
        raise DocumentLoadError("No ECU-700 document chunks found — check data files.")
    if not docs_800:
        raise DocumentLoadError("No ECU-800 document chunks found — check data files.")

    logger.info("ECU-700 chunks: %d | ECU-800 chunks: %d", len(docs_700), len(docs_800))

    # Use unique collection names so repeated calls don't collide in-process
    run_id = uuid.uuid4().hex[:8]
    try:
        store_700 = Chroma.from_documents(
            docs_700, embeddings, collection_name=f"ecu_700_{run_id}"
        )
        store_800 = Chroma.from_documents(
            docs_800, embeddings, collection_name=f"ecu_800_{run_id}"
        )
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to create vector store (embedding API may be "
            f"unavailable or rate-limited): {exc}"
        ) from exc

    raw_700 = "\n\n".join(d.page_content for d in docs_700)
    raw_800 = "\n\n".join(d.page_content for d in docs_800)

    result = {
        "ecu_700": store_700,
        "ecu_800": store_800,
        "raw_docs": {"ecu_700": raw_700, "ecu_800": raw_800},
    }
    _STORE_CACHE["stores"] = result
    logger.info("Vector stores built and cached.")
    return result


def get_retriever(series: str, k: int = 3):
    """Return a LangChain retriever for the specified ECU series.

    Uses k=3: forces the retriever to rank and select the most relevant
    chunks rather than returning everything (ECU-700 has ~4 chunks, so
    k=4 would bypass ranking entirely).

    Args:
        series: Either ``"ECU-700"`` or ``"ECU-800"``.
        k: Number of chunks to return (default 3).

    Returns:
        A LangChain ``VectorStoreRetriever``.

    Raises:
        ValueError: If *series* is not recognised.
    """
    stores = build_vector_stores()
    if "700" in series:
        return stores["ecu_700"].as_retriever(search_kwargs={"k": k})
    if "800" in series:
        return stores["ecu_800"].as_retriever(search_kwargs={"k": k})
    raise ValueError(f"Unknown series identifier: '{series}'")


def get_all_retrievers(k: int = 3) -> dict:
    """Return retrievers for both ECU series.

    Convenience function for cross-series comparison queries.

    Args:
        k: Number of chunks to return per series (default 3).

    Returns:
        Dict with ``"ecu_700"`` and ``"ecu_800"`` retrievers.
    """
    stores = build_vector_stores()
    return {
        "ecu_700": stores["ecu_700"].as_retriever(search_kwargs={"k": k}),
        "ecu_800": stores["ecu_800"].as_retriever(search_kwargs={"k": k}),
    }
