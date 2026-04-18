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


def _retrieve_chunks(retriever, query: str) -> str:
    """Retrieve relevant chunks from the vector store.

    Args:
        retriever: LangChain retriever to query.
        query: The user's natural-language search query.

    Returns:
        Retrieved chunk text joined by double newlines.
    """
    try:
        docs = retriever.invoke(query)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Retriever raised %s: %s",
            type(exc).__name__, exc,
        )
        raise

    if not docs:
        logger.debug("No documents retrieved for query: %s", query)
        return ""

    return "\n\n".join(doc.page_content for doc in docs)


def create_tools(vector_stores: dict) -> list:
    """Create LangChain retrieval tools with injected vector stores.

    Uses closures to bind retriever references into each tool function,
    keeping the tool signatures clean for LangGraph.

    Args:
        vector_stores: Dict returned by :func:`~me_assistant.documents.store.build_vector_stores`.

    Returns:
        List of two :class:`~langchain_core.tools.BaseTool` instances:
        ``[search_ecu_700_docs, search_ecu_800_docs]``.
    """
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
        content = _retrieve_chunks(retriever_700, query)
        return (
            f"[Source: ECU-700_Series_Manual.md]\n"
            f"[IMPORTANT: Only the attributes explicitly mentioned below exist "
            f"in this documentation. If the answer to the user's question is not "
            f"found in the text below, the information is NOT available.]\n\n"
            f"{content}"
        )

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
        content = _retrieve_chunks(retriever_800, query)
        return (
            f"[Source: ECU-800_Series_Base.md, ECU-800_Series_Plus.md]\n"
            f"[IMPORTANT: Only the attributes explicitly mentioned below exist "
            f"in this documentation. If the answer to the user's question is not "
            f"found in the text below, the information is NOT available.]\n\n"
            f"{content}"
        )

    return [search_ecu_700_docs, search_ecu_800_docs]
