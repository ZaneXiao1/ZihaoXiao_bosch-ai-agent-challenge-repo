# LangGraph StateGraph Migration — Task Brief for Claude Code (Minimal Version)

## Context

You are refactoring an existing RAG agent from LangGraph's prebuilt `create_react_agent` to a **custom `StateGraph`** implementation. This is a deliberate architectural upgrade to demonstrate production-grade agent design patterns, not a rewrite of business logic.

The underlying retrieval logic, tools, prompts, and LLM configuration must remain unchanged. Only the **orchestration layer** is being replaced.

## Why this refactor

The current `create_react_agent` setup is a black box — it offers no explicit state, no customization points, and no path to add the production features expected at a senior level:

1. **Explicit state management** — track `iteration_count` and `sources_queried` across nodes instead of hiding them inside the prebuilt executor.
2. **Deterministic safety limits** — enforce a hard ceiling on retrieval loops via graph edges, not by hoping the LLM terminates on its own.
3. **Observability** — named nodes appear as separate spans in LangSmith traces, enabling per-node latency and error analysis.

## Scope boundaries — DO NOT change these

- `src/me_assistant/prompts.py` — keep the existing `SYSTEM_PROMPT` verbatim. It already handles question classification, source selection, citation format, scope discipline, and the "information not in docs" case via natural-language rules.
- `src/me_assistant/tools.py` — keep the existing `@tool`-decorated retrieval functions as-is. Their docstrings drive LLM tool selection.
- `src/me_assistant/loader.py` / `store.py` — unchanged.
- LLM model, temperature, and embedding model — unchanged.

## Architectural decisions — why NOT these nodes

An earlier draft of this architecture included `classify_question`, `grade_retrieval`, and `full_doc_fallback` nodes. **All three are intentionally omitted.** Reasons:

### Why no `classify_question`
The existing `SYSTEM_PROMPT` already instructs the LLM to select tools based on question type (Rule 2/3/4). A pre-classifier duplicates this at extra LLM cost and commits to one interpretation too early — ReAct's strength is letting the LLM see partial results and decide whether to call another tool.

### Why no `grade_retrieval`
In ReAct, **every iteration is an implicit grading step**. When the LLM receives a `ToolMessage`, it decides one of three things: "I have enough, answer now", "I need more, call another tool", or "the context is wrong, retry differently". An explicit grader node just duplicates that decision, adds latency, and provides no signal the LLM doesn't already have. For this corpus size (~15 chunks across 3 files), the duplication has zero accuracy benefit.

### Why no `full_doc_fallback`
The full corpus is ~6KB. Any retrieval that returns at least one chunk already gives the LLM a meaningful fraction of the total available information. "Full document fallback" is a pattern that makes sense when the corpus is 10k+ documents and retrieval failure leaves the LLM with nothing — here it solves a problem that doesn't exist at this scale.

### What replaces them: a retrieval-round ceiling
A hard cap of 3 **retrieval rounds** (tool executions) serves as both a safety valve and an implicit quality signal. If the LLM has consulted the tools 3 times and still wants more, either the documents don't contain the answer (handled by `SYSTEM_PROMPT` Rule 6) or the question is out of scope. A `force_answer` node intercepts when the LLM requests a 4th retrieval round and instructs it to produce its best answer with the context already retrieved, honoring the existing prompt rules about uncertainty.

This simplification is itself a design decision worth highlighting: **adding nodes is easy; knowing when not to add them is the senior-level skill**.

---

## Target architecture

```
START
  ↓
agent (LLM + bind_tools)
  ↓
[conditional edge: did LLM request a tool call?]
  ├─ NO (final answer produced)          → END
  ├─ YES + iteration_count < MAX         → tools → agent (loop)
  └─ YES + iteration_count >= MAX        → force_answer → END
```

Three nodes total: `agent`, `tools`, `force_answer`.

### State schema

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Message history — drives the LLM conversation, accumulates AIMessage/ToolMessage
    messages: Annotated[list, add_messages]

    # Retrieval round counter — incremented by tools_node, read by routing
    iteration_count: int

    # Observability — which sources have been queried; for debugging and eval diagnostics
    sources_queried: list[str]
```

**Semantics of `iteration_count`:** the number of **retrieval rounds already executed**. It is incremented by `tools_node` after a successful tool batch — NOT by `agent_node`. This means `iteration_count` always reflects "how many times have we actually queried the documents". The ceiling `MAX_ITERATIONS = 3` therefore means "at most 3 retrieval rounds", which is the intended user-facing semantic.

`sources_queried` is kept purely for **observability and eval diagnostics** — it records which tools fired. It does not drive any conditional logic.

### Node responsibilities

| Node | Purpose |
|------|---------|
| `agent` | LLM decides next action: call tools, or produce final answer |
| `tools` | Execute the retrieval tool(s) the LLM requested; increment `iteration_count` |
| `force_answer` | Hit only when the LLM requests a retrieval round beyond the ceiling; forces the LLM to answer with context already gathered, then routes to END |

### Conditional edges

**Edge 1: After `agent`**
- If last message has NO `tool_calls` → route to `END` (LLM gave a final answer)
- If last message has `tool_calls` AND `iteration_count < MAX_ITERATIONS` → route to `tools`
- If last message has `tool_calls` AND `iteration_count >= MAX_ITERATIONS` → route to `force_answer`

**Edge 2: After `tools`**
- Unconditional edge back to `agent`

**Edge 3: After `force_answer`**
- Unconditional edge to `END`

### Execution trace (to confirm semantics)

A run that uses all 3 retrieval rounds proceeds as follows:

```
agent #1 → tool_calls → route (count=0, <3) → tools #1 (count becomes 1) → agent #2
agent #2 → tool_calls → route (count=1, <3) → tools #2 (count becomes 2) → agent #3
agent #3 → tool_calls → route (count=2, <3) → tools #3 (count becomes 3) → agent #4
agent #4 → tool_calls → route (count=3, >=3) → force_answer → END
```

At `force_answer`, all 3 rounds of ToolMessages are present in `messages`, so the LLM answers with the maximum retrieved context — no information is wasted. The 4th tool_call request from agent #4 is intentionally discarded; the LLM is instructed to commit to an answer based on what is already available.

---

## File-by-file implementation plan

### File 1: `src/me_assistant/state.py` (new file)

Create the `AgentState` TypedDict exactly as specified above. Add a module docstring explaining the semantic of each field — especially that `iteration_count` counts **retrieval rounds executed**, not agent turns.

### File 2: `src/me_assistant/nodes.py` (new file)

Implement three node functions. Each is a pure function `state -> state_update` — do not mutate state in place, return a dict of fields to update.

#### `agent_node(state: AgentState, llm_with_tools) -> dict`

- Read `messages` from state.
- Prepend `SystemMessage(content=SYSTEM_PROMPT)` ONLY if the first message is not already a `SystemMessage` (otherwise it accumulates every iteration).
- Invoke `llm_with_tools.invoke(messages)`.
- Return `{"messages": [response]}`.

**Do NOT increment `iteration_count` here.** The counter tracks retrieval rounds, not agent turns — it belongs in `tools_node`.

#### `tools_node(state: AgentState, tool_map: dict) -> dict`

This node executes all tool_calls from the last AIMessage and bumps the retrieval counter.

- Read the last message; it must be an `AIMessage` with `tool_calls`.
- For each `tool_call` dict (structure: `{"name": str, "args": dict, "id": str}`):
  - Look up the tool in `tool_map` by `tool_call["name"]`.
  - Execute with `tool.invoke(tool_call["args"])` — **must pass `args` only, not the full tool_call dict**. Passing the whole dict causes the tool to receive unexpected keys and either raise or silently return wrong results.
  - Wrap execution in try/except. On failure, build a `ToolMessage` whose content is the error text — do not crash the graph. The LLM will see the error and can retry or give up gracefully.
  - Build a `ToolMessage(content=result, tool_call_id=tool_call["id"], name=tool_call["name"])`. The `tool_call_id` is required — without it the LLM cannot match the response back to its original request and the next agent turn fails.
  - Record the source for observability: map `"search_ecu_700_docs"` → `"ecu_700"`, `"search_ecu_800_docs"` → `"ecu_800"`. Deduplicate against existing `sources_queried`.
- Return `{"messages": tool_messages, "sources_queried": updated_sources, "iteration_count": state["iteration_count"] + 1}`.

**Key detail — increment exactly once per node invocation, not once per tool_call.** If the LLM requested both `search_ecu_700_docs` and `search_ecu_800_docs` in a single AIMessage, that is **one retrieval round** (one parallel tool batch), counted as one iteration. Incrementing per tool_call would double-count parallel queries and exhaust the budget too fast.

#### `force_answer_node(state: AgentState, llm) -> dict`

Fires when the retrieval ceiling has been reached and the LLM is still requesting tools. Job: terminate the loop gracefully by having the LLM commit to an answer based on already-retrieved context.

- Append a `SystemMessage` to the existing messages with text like:
  > "You have reached the maximum number of retrieval rounds. Based on the information already gathered in this conversation, provide your best answer now. If the retrieved documents do not contain the information needed, explicitly state that following Rule 6 of your instructions — do not request more tool calls."
- Invoke `llm.invoke(messages)` — **use the plain `llm` without `bind_tools`**, so the LLM physically cannot emit `tool_calls`. It must produce a text answer.
- Return `{"messages": [response]}`.

**Why plain `llm` not `llm_with_tools`:** this is a structural guarantee that the graph terminates cleanly. With tools bound, the LLM might request a 4th tool_call that the edge `force_answer → END` would leave unexecuted, producing a broken AIMessage. Removing tools at this node makes termination impossible to violate.

### File 3: `src/me_assistant/graph.py` (REPLACE existing file)

Replace the current `create_react_agent(...)` call with an explicit `StateGraph` construction.

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from .state import AgentState
from .nodes import agent_node, tools_node, force_answer_node
from .tools import search_ecu_700_docs, search_ecu_800_docs

MAX_ITERATIONS = 3  # maximum retrieval rounds


def create_agent(llm):
    tools = [search_ecu_700_docs, search_ecu_800_docs]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)

        if not has_tool_calls:
            # LLM produced final answer
            return END
        if state["iteration_count"] >= MAX_ITERATIONS:
            # Already executed MAX rounds of retrieval; LLM is asking for more — intercept
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

    return workflow.compile()
```

**Critical requirement — initial state.** The MLflow wrapper must pass all three fields explicitly:

```python
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [HumanMessage(content=question)],
    "iteration_count": 0,
    "sources_queried": [],
}
```

Only `messages` has a reducer (`add_messages`). The other fields must be initialized, or nodes that read them will fail with `KeyError`.

### File 4: `src/me_assistant/model.py` (MLflow wrapper — update `predict`)

The existing `MEAssistantModel.predict()` likely calls `self.agent.invoke({"messages": [...]})`. Update it to:

1. Initialize the full state dict with all three fields as shown above.
2. Call `self.agent.invoke(initial_state)`.
3. Extract the final answer from `result["messages"][-1].content`.
4. Optional: expose a `debug` mode that returns `iteration_count` and `sources_queried` — these are what the explicit state gives "for free" and they are valuable for eval.

No changes needed to `load_context()` beyond swapping `create_react_agent` import for `create_agent`.

---

## Testing requirements

Run these checks in order. Do not proceed until the previous passes.

### Check 1: Smoke test — single-source question
```python
result = model.predict(None, {"question": "What is the operating temperature of the ECU-750?"})
```
Expected: correct answer with citation to `ECU-700_Series_Manual.md`. `iteration_count` should be 1 (one retrieval round). `sources_queried` should contain only `"ecu_700"`.

### Check 2: Cross-source comparison
```python
result = model.predict(None, {"question": "Compare the CAN bus capabilities of ECU-750 and ECU-850."})
```
Expected: prose comparison with both sources cited. `sources_queried` should contain both `"ecu_700"` and `"ecu_800"`. `iteration_count` should be 1 if the LLM parallel-called both tools in one round, or 2 if it called them sequentially — both are valid.

### Check 3: Retrieval ceiling
Temporarily set `MAX_ITERATIONS = 1` and run any question. Verify:
- After the first tools execution (count=1), the next agent turn is intercepted by `force_answer` if it requests more tools
- If the LLM produces a final answer directly after round 1, the graph ends via the normal END path (force_answer does NOT fire)
- When force_answer fires, the final response is a coherent text answer (may say "information not available" per Rule 6) — NOT an error, NOT a dangling tool_call

Restore `MAX_ITERATIONS = 3` after verifying.

### Check 4: Off-topic / missing-data case
```python
result = model.predict(None, {"question": "What is the weight of ECU-750?"})  # assuming weight is not in docs
```
Expected: the LLM either responds within iterations that information is not available (Rule 6), OR hits the ceiling and force_answer produces the same outcome. Either way, no fabricated numbers.

### Check 5: Full test suite
Run all 10 questions from `test-questions.csv`. Must achieve ≥ 8/10 correct — same bar as before. Any regression from the previous baseline means the refactor broke something.

---

## Implementation order

Execute in sequence, verifying each before moving on:

1. Create `state.py` — 5 minutes.
2. Create `nodes.py` with all three nodes — 25 minutes (pay attention to tool_call args unpacking and ToolMessage construction).
3. Rewrite `graph.py` — 15 minutes.
4. Update `model.py` predict to pass full state — 10 minutes.
5. Run Check 1 (smoke test) — fix any import/state errors.
6. Run Check 2 (comparison).
7. Run Check 3 (ceiling + force_answer).
8. Run Check 4 (off-topic case).
9. Run Check 5 (full test suite).
10. Update `README.md` architecture section with a mermaid diagram of the 3-node graph.

---

## Common pitfalls to avoid

1. **Do not call `llm.invoke()` in `agent_node`.** Use `llm_with_tools.invoke()`. Without `bind_tools`, the LLM cannot emit tool_calls and the tools node never fires.

2. **Do not forget the `Annotated[list, add_messages]` reducer on `messages`.** Without it, nodes returning `{"messages": [...]}` REPLACE the history instead of appending, breaking the conversation after one turn.

3. **In `tools_node`, pass `tool_call["args"]` to the tool, NOT the whole `tool_call` dict.** The tool expects its actual arguments (e.g., `{"query": "operating temperature"}`), not the wrapping metadata. Pattern:
   ```python
   result = tool_map[tool_call["name"]].invoke(tool_call["args"])
   ```

4. **Every `ToolMessage` must include `tool_call_id=tool_call["id"]`.** Without it, the LLM cannot correlate responses to requests and the next agent turn will fail or produce nonsense.

5. **Increment `iteration_count` in `tools_node`, NOT in `agent_node`.** The counter represents retrieval rounds executed. Incrementing in agent_node creates an off-by-one error where the last round of tool_calls is discarded before execution.

6. **Increment `iteration_count` once per `tools_node` invocation, not once per tool_call.** Parallel tool calls in one AIMessage constitute one retrieval round, not N.

7. **In `force_answer_node`, use plain `llm` WITHOUT `bind_tools`.** This structurally guarantees termination — the LLM physically cannot emit tool_calls, so `force_answer → END` cannot leave a dangling request.

8. **Do not change tool signatures.** Tools currently return strings with source prefixes. Extract source info in `tools_node` from the tool name mapping, not from tool output changes.

9. **Do not re-prepend the SYSTEM_PROMPT on every agent iteration.** Check if `messages[0]` is already a `SystemMessage` before prepending.

---

## Deliverable summary

When complete, the repository should have:
- New files: `src/me_assistant/state.py`, `src/me_assistant/nodes.py`
- Rewritten: `src/me_assistant/graph.py`
- Updated: `src/me_assistant/model.py` (predict method only)
- Unchanged: `prompts.py`, `tools.py`, `loader.py`, `store.py`
- Test pass rate: ≥ 8/10 on `test-questions.csv`, matching or exceeding the prior baseline
- README architecture section updated with a mermaid diagram of the 3-node graph

Report back with:
- Test results for all 10 questions
- For any failures: final `iteration_count`, `sources_queried`, and whether `force_answer` was triggered — these diagnostics are the payoff of the explicit state refactor.
