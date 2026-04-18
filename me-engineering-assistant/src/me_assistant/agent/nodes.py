"""
Node functions for the StateGraph agent.

Each node is a pure function: state -> state_update dict.
No in-place mutation of state.
"""
import logging

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from me_assistant.agent.prompts import SYSTEM_PROMPT
from me_assistant.agent.state import AgentState

logger = logging.getLogger(__name__)

# Maps tool name -> source label for observability
_TOOL_SOURCE_MAP = {
    "search_ecu_700_docs": "ecu_700",
    "search_ecu_800_docs": "ecu_800",
}


def agent_node(state: AgentState, llm_with_tools) -> dict:
    """LLM decides next action: call tools or produce a final answer.

    Prepends the SYSTEM_PROMPT as a SystemMessage only if the first
    message is not already a SystemMessage (avoids accumulation on
    every iteration).

    Returns:
        State update with the LLM response appended to messages.
    """
    messages = list(state["messages"])

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState, tool_map: dict) -> dict:
    """Execute all tool_calls from the last AIMessage and bump the retrieval counter.

    Key details:
    - Passes tool_call["args"] to the tool, NOT the full tool_call dict.
    - Every ToolMessage includes tool_call_id so the LLM can correlate responses.
    - Increments iteration_count exactly ONCE per invocation (not per tool_call).
      Parallel tool calls in one AIMessage = one retrieval round.
    - Records sources for observability, deduplicated against existing sources.

    Returns:
        State update with tool messages, updated sources, and incremented count.
    """
    last_message = state["messages"][-1]
    assert isinstance(last_message, AIMessage) and last_message.tool_calls, (
        "tools_node called but last message has no tool_calls"
    )

    tool_messages = []
    new_sources = list(state.get("sources_queried", []))

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        tool = tool_map.get(tool_name)
        if tool is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Error: unknown tool '{tool_name}'",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
            continue

        try:
            result = tool.invoke(tool_args)
        except Exception as exc:
            logger.error("Tool '%s' raised %s: %s", tool_name, type(exc).__name__, exc)
            result = f"Error executing {tool_name}: {exc}"

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

        # Track source for observability
        source_label = _TOOL_SOURCE_MAP.get(tool_name)
        if source_label and source_label not in new_sources:
            new_sources.append(source_label)

    return {
        "messages": tool_messages,
        "sources_queried": new_sources,
        "iteration_count": state["iteration_count"] + 1,
    }


def force_answer_node(state: AgentState, llm) -> dict:
    """Force the LLM to commit to an answer when the retrieval ceiling is hit.

    Uses the plain LLM (without bind_tools) so it physically cannot emit
    tool_calls, guaranteeing clean termination.

    Returns:
        State update with the forced answer appended to messages.
    """
    messages = list(state["messages"])

    messages.append(
        SystemMessage(
            content=(
                "You have reached the maximum number of retrieval rounds. "
                "Based on the information already gathered in this conversation, "
                "provide your best answer now. If the retrieved documents do not "
                "contain the information needed, explicitly state that following "
                "Rule 6 of your instructions — do not request more tool calls. "
                "CRITICAL REMINDER: Do NOT invent or fabricate any numeric values, "
                "ratings, or specifications that are not explicitly stated in the "
                "retrieved documents. If you cannot quote the exact value from the "
                "documents, say the information is not specified."
            )
        )
    )

    response = llm.invoke(messages)
    return {"messages": [response]}
