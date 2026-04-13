"""Evaluation prompt templates for LLM-as-Judge scoring."""

JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial evaluator for a RAG (Retrieval-Augmented Generation) \
system that answers questions about ECU engineering documentation. \
Your job is to score a model-generated answer on five dimensions, using the \
reference answer, the retrieved context, and the original question as evidence. \
Be rigorous — only award high scores when the answer clearly deserves them.

Return ONLY valid JSON with no additional text, no markdown fencing, no explanation."""

JUDGE_USER_PROMPT = """\
## Question
{question}

## Reference Answer (ground truth)
{reference_answer}

## Evaluation Criteria (from the test set)
{evaluation_criteria}

## Retrieved Context (documents the RAG system actually retrieved)
{retrieved_context}

## Model Answer (to be evaluated)
{model_answer}

---

IMPORTANT — Out-of-Scope and Boundary Questions:
If the question is clearly outside the scope of ECU engineering documentation \
(e.g., weather, jokes, translations, general knowledge, coding requests), the \
model is expected to decline or redirect. When evaluating such questions:
- Any answer that clearly communicates "I can only help with ECU documentation" \
  — whether by explicit refusal ("I cannot answer that") or by redirecting \
  ("I'm here to assist with ECU specifications") — should be treated as a \
  CORRECT and COMPLETE response. Both styles are equally acceptable.
- For these out-of-scope questions, score all five dimensions based on how \
  well the model declines, NOT on whether the wording matches the reference \
  answer exactly. A polite redirect scores the same as an explicit refusal.
- Faithfulness for out-of-scope questions: if no context was retrieved (which \
  is expected), and the model simply declines/redirects without making factual \
  claims, score faithfulness 5/5 — the model made no unsupported claims.

Score the Model Answer on the following five dimensions. For each dimension, \
provide an integer score (0-5) and a brief rationale (1-2 sentences).

### 1. Correctness (0-5)
How factually accurate is the model answer compared to the reference answer?
- **5**: All facts, numbers, units, and technical details match the reference exactly.
- **4**: Core facts are correct; at most one minor imprecision (e.g., rounding, \
  wording variation) that does not change the meaning.
- **3**: The main point is correct, but one or two secondary facts are wrong or \
  imprecise.
- **2**: The answer contains the right topic but gets important specifics wrong \
  (e.g., wrong value, wrong model attribution).
- **1**: Mostly incorrect, but shows some awareness of the relevant domain.
- **0**: Completely wrong, fabricated, or refuses to answer when the information \
  is available.

### 2. Completeness (0-5)
Does the model answer cover all the key information in the reference answer?
- **5**: Every key point in the reference is present; nothing meaningful is missing.
- **4**: Covers all major points; at most one minor detail is absent.
- **3**: Covers the main point but misses one significant piece of information \
  from the reference.
- **2**: Answers only part of the question; multiple important points from the \
  reference are missing.
- **1**: Barely addresses the question; most reference content is absent.
- **0**: Does not address the question at all, or returns an empty / irrelevant answer.

### 3. Faithfulness (0-5)
Does the model answer ONLY state things that are supported by the Retrieved Context \
shown above? This dimension measures groundedness — whether every claim in the \
answer can be traced back to the retrieved documents. Do NOT compare against the \
reference answer for this dimension; only check against the Retrieved Context.
- **5**: Every claim in the answer is directly supported by the retrieved context; \
  zero hallucination.
- **4**: All substantive claims are supported by the context; minor stylistic \
  phrasing that does not introduce false facts.
- **3**: Mostly grounded in the context, but includes one claim that cannot be \
  found in the retrieved documents.
- **2**: Contains one or more clearly unsupported claims — information that is \
  not present in the retrieved context.
- **1**: Significant hallucination — multiple claims have no basis in the \
  retrieved context.
- **0**: Entirely fabricated or contradicts the retrieved context on every key point.

### 4. Relevance / Scope Control (0-5)
Does the answer stay focused on exactly what the user asked, without adding \
unrequested information?
- **5**: Answers precisely what was asked — no extra specifications, no \
  unrequested model comparisons, no tangential details.
- **4**: Focused on the question; at most one minor extra detail that does not \
  distract.
- **3**: Mostly on-topic but includes one clearly unrequested specification or \
  introduces a model the user did not ask about.
- **2**: Contains multiple unrequested details or specifications that dilute the \
  answer.
- **1**: The answer wanders significantly — more unrequested content than relevant \
  content.
- **0**: Does not address the user's question at all, or is entirely off-topic.

### 5. Format Compliance (0-5)
Does the model answer follow the system prompt formatting rules?
The rules are:
- Use prose only (no tables, no bullet lists, no markdown headers).
- Cite source documents (e.g., "(source: ECU-700_Series_Manual.md)").
- Single-fact questions: short and concise (roughly 1-3 sentences).
- Comparison questions: prose contrasting the attributes + a brief analysis sentence.

NOTE: Minor length deviations that do not harm clarity should not be penalized \
heavily. Focus on structural violations (tables, bullets, headers, missing \
citations) rather than exact sentence counts.

Scoring:
- **5**: Fully compliant — prose format, sources cited, appropriate structure.
- **4**: One minor deviation (e.g., slightly verbose but still clear prose, or \
  citation format slightly varies).
- **3**: One notable structural issue (e.g., uses a bullet list but content is \
  correct, or missing source citation entirely).
- **2**: Multiple structural violations (e.g., uses a table AND missing citations).
- **1**: Major format violations — ignores most formatting rules.
- **0**: Completely ignores the format (e.g., dumps raw retrieved text, uses \
  headers + tables + bullets throughout).

---

Return your evaluation as JSON with exactly this structure:
{{
  "correctness": {{"score": <int 0-5>, "rationale": "<string>"}},
  "completeness": {{"score": <int 0-5>, "rationale": "<string>"}},
  "faithfulness": {{"score": <int 0-5>, "rationale": "<string>"}},
  "relevance": {{"score": <int 0-5>, "rationale": "<string>"}},
  "format_compliance": {{"score": <int 0-5>, "rationale": "<string>"}}
}}"""
