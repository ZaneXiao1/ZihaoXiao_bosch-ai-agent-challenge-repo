"""
Agent state schema for the LangGraph StateGraph.

Field semantics:
- messages: The LLM conversation history. Uses the ``add_messages`` reducer
  so that node returns *append* to the list rather than replacing it.
- iteration_count: Number of **retrieval rounds executed** (not agent turns).
  Incremented by ``tools_node`` after each tool batch. A single AIMessage
  that requests two parallel tool calls counts as one retrieval round.
- sources_queried: Observability / eval diagnostics only. Records which
  tool names have fired (deduplicated). Does NOT drive conditional logic.
"""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for the custom StateGraph agent."""

    messages: Annotated[list, add_messages]
    iteration_count: int
    sources_queried: list[str]
