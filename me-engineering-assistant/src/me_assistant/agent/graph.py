"""
LangGraph ReAct agent for ECU engineering documentation queries.

Design Decision: The ReAct pattern lets the LLM autonomously decide which
tool(s) to call based on the question — single tool for focused queries,
both tools for cross-series comparisons.  This routing is pure language
understanding (Layer 1); each tool then performs embedding similarity search
inside its own FAISS index (Layer 2).
"""
import logging
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from me_assistant.agent.prompts import SYSTEM_PROMPT
from me_assistant.agent.tools import create_tools
from me_assistant.config import get_llm
from me_assistant.documents.store import build_vector_stores
from me_assistant.exceptions import (
    ConfigurationError,
    EmbeddingError,
    LLMError,
    MEAssistantError,
    RetrievalError,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for the LangGraph ReAct agent."""

    messages: Annotated[list, add_messages]


def create_agent():
    """Create and return the compiled LangGraph ReAct agent.

    Builds vector stores once (cached after the first call), creates the
    retrieval tools with injected store references, and wires the LangGraph
    graph with the configured LLM.

    Returns:
        A compiled ``CompiledGraph`` ready to invoke with
        ``{"messages": [("user", question)]}``.

    Raises:
        ConfigurationError: If provider config or API keys are invalid.
        EmbeddingError: If the embedding model fails during vector store creation.
        RetrievalError: If vector store construction fails.
    """
    logger.info("Initialising ME Engineering Assistant agent...")

    try:
        vector_stores = build_vector_stores()
    except MEAssistantError:
        raise
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to build vector stores: {exc}"
        ) from exc

    tools = create_tools(vector_stores)

    try:
        llm = get_llm()
    except MEAssistantError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to initialise LLM: {exc}"
        ) from exc

    agent = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
    )
    logger.info("Agent created successfully.")
    return agent


def query_agent(agent, question: str) -> str:
    """Run a single query through the agent and return the answer string.

    Args:
        agent: Compiled LangGraph agent returned by :func:`create_agent`.
        question: The user's natural-language question.

    Returns:
        The agent's final answer as a plain string.

    Raises:
        LLMError: If the LLM API call fails (auth, rate limit, timeout, etc.).
        RetrievalError: If tool invocation fails during retrieval.
    """
    try:
        result = agent.invoke({"messages": [("user", question)]})
    except MEAssistantError:
        raise
    except Exception as exc:
        exc_name = type(exc).__name__
        exc_module = type(exc).__module__ or ""

        # Detect common LLM API errors by class name / module so we don't
        # need a hard import on openai (which may not be installed).
        if any(kw in exc_name for kw in ("Auth", "Permission")):
            raise LLMError(f"LLM authentication failed: {exc}") from exc
        if "RateLimit" in exc_name:
            raise LLMError(
                f"LLM rate limit exceeded — retry later: {exc}"
            ) from exc
        if any(kw in exc_name for kw in ("Timeout", "TimeoutError")):
            raise LLMError(f"LLM request timed out: {exc}") from exc
        if "openai" in exc_module or "anthropic" in exc_module:
            raise LLMError(f"LLM API error ({exc_name}): {exc}") from exc

        raise LLMError(
            f"Agent invocation failed ({exc_name}): {exc}"
        ) from exc

    return result["messages"][-1].content
