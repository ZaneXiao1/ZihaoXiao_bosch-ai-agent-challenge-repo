# Quick Reference: Agent Architecture

## For Interviews (1-minute explanation)

**"How does your agent work?"**

> "It's a ReAct agent—LLM thinks, decides which tools to call, observes results, and thinks again. But I made it production-grade with three enhancements:
> 1. **Explicit state** — `iteration_count` and `sources_queried` track progress
> 2. **Hard safety limit** — routing blocks at iteration 3 (structurally impossible to loop more)
> 3. **Per-node observability** — each node (agent/tools/force_answer) is a separate LangSmith span
> 
> So unlike `create_react_agent`, I have deterministic cost control and clear visibility into what's happening."

---

## Core Loop

```
[1. Agent Thinks]
    ↓ (sees full conversation)
[2. Agent Decides]
    ├─ "I have answer" → no tool_calls
    │   ↓
    └─→ RETURN ANSWER
    │
    └─ "I need tools" + iter < 3
    │   ↓
    └─→ EXECUTE TOOLS → increment iteration_count → loop back
    │
    └─ "I need tools" + iter ≥ 3
        ↓
        → FORCE_ANSWER (no tools) → RETURN ANSWER
```

---

## Three Key Components

| Component | Role | Key Feature |
|---|---|---|
| **Agent Node** | LLM reads tools, decides action | Sees FULL conversation before deciding |
| **Tools Node** | Executes retrieval, increments counter | Parallel tools = 1 iteration (not N) |
| **Force Answer** | LLM without tools | Structurally prevents further loops |

---

## Key Metrics

| Metric | Meaning | Example |
|---|---|---|
| `iteration_count` | How many retrieval rounds | 1 (simple), 2 (multi-part), 3 (ceiling) |
| `sources_queried` | Which tools fired | ["ecu_700"], ["ecu_700", "ecu_800"] |
| `messages` | Full conversation | [HumanMessage, AIMessage, ToolMessage, ...] |

---

## ReAct Comparison

| Aspect | Pure ReAct | Your Design |
|---|---|---|
| **Pattern** | Reason→Act→Observe loop | ✅ Same + state management |
| **Iteration limit** | Soft (prompt) | Hard (routing block) |
| **Cost predictable?** | No | Yes (max 3 rounds) |
| **Tool tracking** | Must parse | Explicit sources_queried |
| **Per-node visibility** | No | Yes (LangSmith spans) |

---

## Decision Tree

```
Agent emits tool_calls?
├─ NO → return answer (done)
└─ YES
   ├─ iteration_count < 3 → execute tools → loop back
   └─ iteration_count ≥ 3 → force_answer (no tools) → return
```

---

## When to Use Which

- **Simple lookup** (1 iteration) — "What is the max temp?"
- **Cross-series** (1 iteration, parallel) — "Compare ECU-750 vs ECU-850?"
- **Multi-part** (2 iterations) — "What's the temp, power, and I/O?"
- **Vague query** (hits ceiling) — "Tell me everything" → force_answer

---

## Why This Design?

1. **Deterministic** — Can't loop more than 3 times, no exceptions
2. **Observable** — See exactly which tools fired when
3. **Cost-predictable** — Know max token spend upfront
4. **Production-ready** — Explicit state for debugging
5. **ReAct-based** — Proven agent pattern, just enhanced

---

## Quick Links

- **High-level overview** — README.md § 1.2
- **Detailed execution traces** — docs/agent_graph_examples.md
- **ReAct analysis** — docs/react_vs_custom_design.md

