# Agent Graph Architecture - Detailed Examples

This document provides complete, step-by-step execution traces for different query scenarios. Refer to this when understanding how the agent loop works in practice.

---

## Example 1: Simple Lookup (1 Iteration)

**Query:** "What is the maximum operating temperature for the ECU-750?"

```
ENTRY POINT:
  messages = [HumanMessage("What is the max temp for ECU-750?")]
  iteration_count = 0
  sources_queried = []

┌─────────────────────────────────────────────────────────────────────────────┐
│ ITERATION 1 — First Decision                                               │
└─────────────────────────────────────────────────────────────────────────────┘

[AGENT NODE RUNS]
  Input state:
    messages = [HumanMessage("What is the max temp for ECU-750?")]
    iteration_count = 0
  
  Processing:
    - LLM receives system prompt + conversation
    - LLM reads tool docstrings to understand what's available
    - LLM decides: "I need to search ECU-700 docs for ECU-750 specs"
    - LLM emits: tool_calls = [{"name": "search_ecu_700_docs", ...}]
  
  State after agent_node:
    messages = [HumanMessage(...), AIMessage(tool_calls=[...])]
    iteration_count = 0  # Not incremented yet!

[ROUTING DECISION]
  Check 1: Does last AIMessage have tool_calls?
    → YES (has tool_calls)
  
  Check 2: Is iteration_count (0) < MAX_ITERATIONS (3)?
    → YES (0 < 3)
  
  Decision: ROUTE TO "tools" NODE

[TOOLS NODE RUNS]
  Processing:
    - Execute search_ecu_700_docs: "maximum temperature ECU-750"
    - Result: "[ECU-700 | ECU-750] Specifications... operates between -40°C and +85°C..."
    - Wrap in ToolMessage with original tool_call_id
    - Increment iteration_count: 0 → 1
    - Add "ecu_700" to sources_queried
  
  State after tools_node:
    messages = [HumanMessage(...), AIMessage(tool_calls=[...]), ToolMessage(...)]
    iteration_count = 1  # INCREMENTED HERE
    sources_queried = ["ecu_700"]

[ROUTING DECISION]
  Last message is ToolMessage (not AIMessage)
  → ROUTE BACK TO "agent" NODE

┌─────────────────────────────────────────────────────────────────────────────┐
│ ITERATION 2 — Second Decision (Agent Re-evaluates)                         │
└─────────────────────────────────────────────────────────────────────────────┘

[AGENT NODE RUNS AGAIN]
  Input state:
    messages = [
      HumanMessage("What is the max temp for ECU-750?"),
      AIMessage(tool_calls=[...]),
      ToolMessage(content="[ECU-700 | ECU-750]... -40°C to +85°C...")  # NEW
    ]
    iteration_count = 1
  
  Processing:
    - LLM sees FULL conversation including the retrieved context
    - LLM reads: "-40°C to +85°C" in the tool result
    - LLM decides: "I have enough information to answer now"
    - LLM emits: NO tool_calls, just the final answer
  
  State after agent_node (2nd run):
    messages = [..., AIMessage(content="The ECU-750 operates between...")]
    iteration_count = 1  # Unchanged
    sources_queried = ["ecu_700"]

[ROUTING DECISION]
  Check: Does last AIMessage have tool_calls?
    → NO (tool_calls is empty)
  
  Decision: ROUTE TO END (exit the graph)

┌─────────────────────────────────────────────────────────────────────────────┐
│ FINAL RESULT                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Returned to user:
  answer = "The ECU-750 operates between -40°C and +85°C according to the specifications."
  iteration_count = 1  (only 1 retrieval round was needed)
  sources_queried = ["ecu_700"]
  messages = [full conversation history with all turns]
```

---

## Example 2: Cross-Series Comparison (1 Iteration, Parallel Tools)

**Query:** "Compare OTA update capabilities across ECU-750 and ECU-850"

Key point: Agent calls **both tools in parallel**, which counts as a single iteration.

```
[AGENT NODE]
  Decision: "This question asks about TWO different ECU series.
             I need to call both search_ecu_700_docs AND search_ecu_800_docs"
  
  Output: AIMessage(tool_calls=[
    {"name": "search_ecu_700_docs", "args": {"query": "OTA update capability ECU-750"}},
    {"name": "search_ecu_800_docs", "args": {"query": "OTA update capability ECU-850"}}
  ])
  
  iteration_count = 0

[TOOLS NODE]
  Processing:
    ✓ Tool call 1 (parallel): search_ecu_700_docs → "OTA updates are NOT supported on ECU-750"
    ✓ Tool call 2 (parallel): search_ecu_800_docs → "ECU-850 supports OTA updates via WiFi"
    ✓ Both execute simultaneously (same iteration)
    ✓ Increment iteration_count: 0 → 1 (only ONCE!)
    ✓ Update sources: ["ecu_700", "ecu_800"]
  
  State after tools_node:
    messages = [
      HumanMessage(...),
      AIMessage(tool_calls=[...]),
      ToolMessage(content="...[ECU-700]... OTA NOT supported...", tool_call_id=...),
      ToolMessage(content="...[ECU-800]... OTA supported via WiFi...", tool_call_id=...)
    ]
    iteration_count = 1  # Single increment despite 2 tool calls!
    sources_queried = ["ecu_700", "ecu_800"]

[AGENT NODE (2nd run)]
  Input: Full conversation including both tool results
  
  Decision: "I now have both ECU-700 and ECU-800 information. I can synthesize the comparison."
  
  Output: AIMessage(
    content="ECU-750 does NOT support OTA updates, 
             while ECU-850 supports OTA updates via WiFi connection.",
    tool_calls=[]  # No more tools needed
  )

[ROUTING]
  Check: tool_calls in AIMessage?
    → NO
  
  ROUTE TO END

RESULT:
  iteration_count = 1 (completed in 1 retrieval round, despite 2 parallel tool calls)
  sources_queried = ["ecu_700", "ecu_800"]
```

---

## Example 3: Multi-Part Question (2 Iterations)

**Query:** "What is the ECU-850's maximum operating temperature, power consumption, and I/O options?"

Key point: Agent decides it needs a second search to complete the answer.

```
[AGENT NODE - Round 1]
  Decision: "I need multiple attributes of ECU-850. Let me start with a broad search."
  
  Output: AIMessage(tool_calls=[
    {"name": "search_ecu_800_docs", "args": {"query": "ECU-850 temperature power consumption I/O"}}
  ])
  
  iteration_count = 0

[TOOLS NODE - Round 1]
  Result: "[ECU-800 | ECU-850] Specifications... Temp: -40°C to +85°C... Power: 2.5W..."
          (but missing I/O details)
  
  iteration_count: 0 → 1
  sources_queried: ["ecu_800"]

[AGENT NODE - Round 2]
  Input: Full conversation including the partial result
  
  Review: "I found temperature and power, but the I/O options are NOT in the returned text."
  
  Decision: "Need to refine search for I/O details specifically"
  
  Output: AIMessage(tool_calls=[
    {"name": "search_ecu_800_docs", "args": {"query": "ECU-850 I/O interface inputs outputs connections"}}
  ])
  
  iteration_count = 1  # Still 1 from previous iteration

[TOOLS NODE - Round 2]
  Result: "[ECU-800 | ECU-850] I/O... CAN interface, Ethernet, Digital I/O (16 channels)..."
  
  iteration_count: 1 → 2  # Now incremented to 2
  sources_queried: ["ecu_800"]  # Still just one source

[AGENT NODE - Round 3]
  Input: Full conversation with BOTH previous queries and results
  
  Review: "Now I have:
           - Temperature: -40°C to +85°C ✓
           - Power: 2.5W ✓
           - I/O: CAN, Ethernet, 16 Digital I/O ✓
           All parts are answered."
  
  Output: AIMessage(
    content="The ECU-850 has the following specifications:
             - Maximum operating temperature: -40°C to +85°C
             - Power consumption: 2.5W
             - I/O options: CAN interface, Ethernet, 16 Digital I/O channels",
    tool_calls=[]  # No more tools needed
  )

[ROUTING]
  Check: tool_calls in AIMessage?
    → NO
  
  ROUTE TO END

RESULT:
  iteration_count = 2 (required 2 retrieval rounds)
  sources_queried = ["ecu_800"]
  The Agent autonomously decided it needed a second search to complete the answer
```

---

## Example 4: Safety Ceiling in Action (3 Iterations)

**Query:** "Tell me everything about every ECU model" (vague, impossible to fully answer)

Key point: At iteration 3, the routing logic blocks further tool calls and forces an answer.

```
[AGENT NODE - Round 1]
  Decision: "Searching for comprehensive ECU information"
  
  Output: AIMessage(tool_calls=[...])
  iteration_count = 0

[TOOLS NODE - Round 1]
  Execute searches
  iteration_count: 0 → 1

[AGENT NODE - Round 2]
  Thinks: "I have ECU-700 info, but need more comprehensive coverage"
  
  Output: AIMessage(tool_calls=[...])
  iteration_count = 1

[TOOLS NODE - Round 2]
  Execute searches
  iteration_count: 1 → 2

[AGENT NODE - Round 3]
  Thinks: "Still want more details..."
  
  Output: AIMessage(tool_calls=[...])
  iteration_count = 2

[TOOLS NODE - Round 3]
  Execute searches
  iteration_count: 2 → 3

[AGENT NODE - Round 4 (WANTS MORE)]
  Thinks: "I should search for even more information..."
  
  Decision: Emit tool_calls again
  
  Output: AIMessage(tool_calls=[
    {"name": "search_ecu_700_docs", ...},
    {"name": "search_ecu_800_docs", ...}
  ])
  
  iteration_count = 3  # STILL 3! (tools_node hasn't run yet)

[ROUTING DECISION - CRITICAL]
  Check 1: Does AIMessage have tool_calls?
    → YES (Agent wants more tools)
  
  Check 2: Is iteration_count (3) < MAX_ITERATIONS (3)?
    → FALSE!!! (3 is NOT < 3)
    
    ⚠️  AGENT WANTS MORE TOOLS BUT WE'VE HIT THE CEILING ⚠️
  
  Decision: ROUTE TO "force_answer" NODE (not "tools")
           This PREVENTS the tools_node from running

[FORCE_ANSWER NODE]
  Input: Full conversation with 3 retrieval rounds already completed
  
  Processing:
    - Create plain LLM WITHOUT bind_tools()
    - LLM cannot generate tool_calls (no tools available)
    - Prepend system message: "You have reached the maximum retrieval rounds..."
    - Force LLM to synthesize answer from existing context
  
  Output: AIMessage(
    content="Based on the available ECU documentation, I can provide:
             [summarizes everything from 3 rounds of searches]
             For additional details not covered here, the documentation
             does not provide further information.",
    tool_calls=[]  # GUARANTEED EMPTY (LLM has no tools)
  )

[ROUTING]
  Check: tool_calls in AIMessage?
    → NO (cannot be present)
  
  ROUTE TO END

RESULT:
  iteration_count = 3  (ceiling enforced)
  sources_queried = ["ecu_700", "ecu_800"]
  Answer provided despite vague question, loop terminated cleanly
```

### Key Insight

The ceiling is enforced by the **routing decision BEFORE tools_node runs**. The Agent's desire to call tools again is blocked at the graph level, not at the LLM level. This is a **structural guarantee**, not a behavioral hope.

---

## Routing Decision Flowchart

After every `agent_node` execution, the routing logic checks:

```
                    Agent finishes
                          │
                          ▼
           ┌──────────────────────────────┐
           │ Last AIMessage has          │
           │ tool_calls?                 │
           └──┬──────────────────────┬───┘
              │                      │
           NO │                      │ YES
              │                      │
              ▼                      ▼
          ┌──────────┐   ┌──────────────────────┐
          │ ROUTE    │   │ iteration_count >= 3?│
          │  END     │   └──┬──────────────┬────┘
          │          │      │              │
          └──────────┘   NO │              │ YES
                           │              │
                           ▼              ▼
                      ┌─────────────┐ ┌──────────────┐
                      │ ROUTE       │ │ ROUTE        │
                      │ "tools"     │ │ "force_      │
                      │             │ │  answer"     │
                      └─────────────┘ └──────────────┘
```

The three possible outcomes:

| Condition | Action | Result |
|---|---|---|
| Agent says "I'm done" (no tool_calls) | ROUTE END | Return answer immediately |
| Agent wants tools AND iter < 3 | ROUTE tools | Execute, increment, loop back to agent |
| Agent wants tools BUT iter >= 3 | ROUTE force_answer | Generate answer without tools |
