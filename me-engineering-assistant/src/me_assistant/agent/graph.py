"""
Custom LangGraph StateGraph agent for ECU engineering documentation queries.

Architecture: Three nodes — agent, tools, force_answer — with explicit state
tracking for iteration_count and sources_queried. Replaces the prebuilt
create_react_agent with a production-grade StateGraph that provides:
1. Explicit state management across retrieval rounds
2. Deterministic safety limits via a hard retrieval ceiling
3. Named-node observability for per-node tracing in LangSmith
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from me_assistant.agent.nodes import agent_node, force_answer_node, tools_node
from me_assistant.agent.state import AgentState
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

MAX_ITERATIONS = 3  # maximum retrieval rounds before forcing an answer


def create_agent():
    """Create and return the compiled StateGraph agent.

    Builds vector stores once (cached after the first call), creates the
    retrieval tools with injected store references, and wires the StateGraph
    with the configured LLM.

    Returns:
        A compiled ``CompiledGraph`` ready to invoke with the full
        AgentState dict (messages, iteration_count, sources_queried).

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

    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def route_after_agent(state: AgentState) -> str:
        """Conditional edge after the agent node."""
        last = state["messages"][-1]
        has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)

        if not has_tool_calls:
            return END
        if state["iteration_count"] >= MAX_ITERATIONS:
            return "force_answer"
        return "tools"

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", lambda s: agent_node(s, llm_with_tools))
    workflow.add_node("tools", lambda s: tools_node(s, tool_map))
    workflow.add_node("force_answer", lambda s: force_answer_node(s, llm))

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "force_answer": "force_answer", END: END},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("force_answer", END)

    agent = workflow.compile()
    logger.info("Agent created successfully.")
    return agent


def query_agent(agent, question: str) -> dict:
    """Run a single query through the agent and return the full result state.

    Args:
        agent: Compiled StateGraph agent returned by :func:`create_agent`.
        question: The user's natural-language question.

    Returns:
        Dict with keys: answer (str), iteration_count (int),
        sources_queried (list[str]), messages (list).

    Raises:
        LLMError: If the LLM API call fails (auth, rate limit, timeout, etc.).
        RetrievalError: If tool invocation fails during retrieval.
    """
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration_count": 0,
        "sources_queried": [],
    }

    try:
        result = agent.invoke(initial_state)
    except MEAssistantError:
        raise
    except Exception as exc:
        exc_name = type(exc).__name__
        exc_module = type(exc).__module__ or ""

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

    return {
        "answer": result["messages"][-1].content,
        "iteration_count": result["iteration_count"],
        "sources_queried": result["sources_queried"],
        "messages": result["messages"],
    }
