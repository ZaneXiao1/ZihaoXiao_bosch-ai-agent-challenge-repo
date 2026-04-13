"""
Retrieval tools for the LangGraph agent.

Design Decision: Separate tools per document series allows the agent
to intelligently decide which documentation to consult. For cross-series
comparisons, the agent calls both tools.

CRITICAL: The tool docstrings are what the LLM reads to decide which tool
to call. They must clearly describe what each series covers and when to use
each tool. Poor docstrings = wrong routing = wrong answers.
"""
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _retrieve_with_fallback(retriever, full_doc_content: str, query: str) -> str:
    """Retrieve relevant chunks, falling back to full document if quality is low.

    Since the total corpus is only ~6 KB, passing the full document as context
    costs ~1 500 tokens — well within any LLM context window. This guarantees
    correctness even when a relevant chunk scores poorly in embedding similarity
    (e.g. "OTA not supported" vs. "which supports OTA").

    Args:
        retriever: LangChain retriever to query.
        full_doc_content: Full concatenated document text for fallback use.
        query: The user's natural-language search query.

    Returns:
        Retrieved chunk text, or full document content when retrieval quality
        is insufficient.
    """
    try:
        docs = retriever.invoke(query)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Retriever raised %s, using full-document fallback: %s",
            type(exc).__name__, exc,
        )
        return full_doc_content

    if not docs:
        logger.debug("No documents retrieved — returning full document fallback.")
        return full_doc_content

    retrieved_text = "\n\n".join(doc.page_content for doc in docs)

    if len(retrieved_text) < 200:
        logger.debug(
            "Retrieved text is only %d chars — returning full document fallback.",
            len(retrieved_text),
        )
        return full_doc_content

    return retrieved_text


def create_tools(vector_stores: dict) -> list:
    """Create LangChain retrieval tools with injected vector stores.

    Uses closures to bind retriever and raw-doc references into each tool
    function, keeping the tool signatures clean for LangGraph.

    Args:
        vector_stores: Dict returned by :func:`~me_assistant.documents.store.build_vector_stores`.

    Returns:
        List of two :class:`~langchain_core.tools.BaseTool` instances:
        ``[search_ecu_700_docs, search_ecu_800_docs]``.
    """
    raw_700: str = vector_stores["raw_docs"]["ecu_700"]
    raw_800: str = vector_stores["raw_docs"]["ecu_800"]
    retriever_700 = vector_stores["ecu_700"].as_retriever(search_kwargs={"k": 3})
    retriever_800 = vector_stores["ecu_800"].as_retriever(search_kwargs={"k": 3})

    @tool
    def search_ecu_700_docs(query: str) -> str:
        """Search the ECU-700 Series documentation (legacy product line, covers ECU-750).

        Use this tool when the question involves ANY of the following:
        - The ECU-700 series or ECU-750 model specifically
        - Legacy/older ECU product specifications
        - Comparing legacy vs newer ECU models (use BOTH this tool and search_ecu_800_docs)
        - Questions about which models support/don't support a feature (must check ALL series)

        Args:
            query: The search query about ECU-700 series specifications.

        Returns:
            Relevant documentation excerpts from ECU-700 series manuals.
        """
        content = _retrieve_with_fallback(retriever_700, raw_700, query)
        return f"[Source: ECU-700_Series_Manual.md]\n{content}"

    @tool
    def search_ecu_800_docs(query: str) -> str:
        """Search the ECU-800 Series documentation (next-gen, covers ECU-850 and ECU-850b).

        Use this tool when the question involves ANY of the following:
        - The ECU-800 series, ECU-850 model, or ECU-850b model
        - Next-generation ECU features and specifications
        - Comparing ECU-850 vs ECU-850b (both are in this documentation)
        - Comparing newer vs legacy ECU models (use BOTH this tool and search_ecu_700_docs)
        - Questions about which models support/don't support a feature (must check ALL series)

        Args:
            query: The search query about ECU-800 series specifications.

        Returns:
            Relevant documentation excerpts from ECU-800 series manuals.
        """
        content = _retrieve_with_fallback(retriever_800, raw_800, query)
        return f"[Source: ECU-800_Series_Base.md, ECU-800_Series_Plus.md]\n{content}"

    return [search_ecu_700_docs, search_ecu_800_docs]
