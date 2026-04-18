"""
LLM-as-Judge evaluator for RAG answer quality.

Loads test cases from CSV, runs each question through the agent,
extracts retrieved context from tool messages, calls a judge LLM to
score the answer on 5 dimensions, and records per-question results
including latency.
"""
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from me_assistant.config import get_llm
from me_assistant.evaluation.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT

def _get_judge_llm():
    """Return a dedicated judge LLM (gpt-4o) separate from the agent's model."""
    return ChatOpenAI(model="gpt-4o", temperature=0)

logger = logging.getLogger(__name__)

DEFAULT_TEST_SET = Path(__file__).parent.parent.parent.parent / "tests" / "test_queries.csv"

DIMENSIONS = ("correctness", "completeness", "faithfulness", "relevance", "format_compliance")


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    score: int
    rationale: str


@dataclass
class EvalResult:
    """Full evaluation result for one test case."""
    question_id: int
    category: str
    question: str
    reference_answer: str
    model_answer: str
    retrieved_context: str = ""
    correctness: Optional[DimensionScore] = None
    completeness: Optional[DimensionScore] = None
    faithfulness: Optional[DimensionScore] = None
    relevance: Optional[DimensionScore] = None
    format_compliance: Optional[DimensionScore] = None
    latency_seconds: float = 0.0
    judge_error: str = ""

    @property
    def total_score(self) -> int:
        return sum(
            getattr(self, d).score
            for d in DIMENSIONS
            if getattr(self, d) is not None
        )

    @property
    def max_possible(self) -> int:
        return len(DIMENSIONS) * 5  # 5 dimensions * 5 max each = 25

    def to_dict(self) -> dict:
        d = {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "model_answer": self.model_answer,
            "retrieved_context": self.retrieved_context,
            "latency_seconds": round(self.latency_seconds, 2),
            "judge_error": self.judge_error,
            "total_score": self.total_score,
            "max_possible": self.max_possible,
        }
        for dim_name in DIMENSIONS:
            dim = getattr(self, dim_name)
            if dim:
                d[f"{dim_name}_score"] = dim.score
                d[f"{dim_name}_rationale"] = dim.rationale
            else:
                d[f"{dim_name}_score"] = None
                d[f"{dim_name}_rationale"] = None
        return d


def load_test_cases(csv_path: Optional[str] = None) -> list[dict]:
    """Load test cases from CSV file.

    Returns a list of dicts with keys:
        question_id, category, question, expected_answer, evaluation_criteria
    """
    path = Path(csv_path) if csv_path else DEFAULT_TEST_SET
    cases = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append({
                "question_id": int(row["Question_ID"]),
                "category": row["Category"],
                "question": row["Question"],
                "expected_answer": row["Expected_Answer"],
                "evaluation_criteria": row.get("Evaluation_Criteria", ""),
            })
    return cases


def _extract_retrieved_context(messages: list) -> str:
    """Extract retrieved context from ToolMessage entries in agent output.

    In the ReAct agent, every tool call produces a ToolMessage whose content
    is the text returned by the retrieval tool. We concatenate all of them
    to form the full retrieved context the model had access to.
    """
    from langchain_core.messages import ToolMessage

    contexts = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            contexts.append(msg.content)
    return "\n\n---\n\n".join(contexts) if contexts else "[No context retrieved]"


def _parse_judge_response(text: str) -> dict:
    """Parse the judge LLM's JSON response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


def judge_answer(
    question: str,
    reference_answer: str,
    evaluation_criteria: str,
    retrieved_context: str,
    model_answer: str,
    judge_llm=None,
) -> dict:
    """Call the judge LLM to score a single model answer.

    Args:
        question: The original question.
        reference_answer: Ground-truth answer from the test set.
        evaluation_criteria: Human-defined criteria for this question.
        retrieved_context: The documents actually retrieved by the RAG system.
        model_answer: The RAG system's generated answer.
        judge_llm: LangChain LLM to use as judge. If None, creates one via config.

    Returns:
        Parsed dict with scores for all 5 dimensions.
    """
    if judge_llm is None:
        judge_llm = _get_judge_llm()

    prompt = JUDGE_USER_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        evaluation_criteria=evaluation_criteria,
        retrieved_context=retrieved_context,
        model_answer=model_answer,
    )

    messages = [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", prompt),
    ]
    response = judge_llm.invoke(messages)
    return _parse_judge_response(response.content)


def _query_agent_with_context(agent, question: str) -> tuple[str, str]:
    """Run a query and return (final_answer, retrieved_context).

    Invokes the agent with the full AgentState (messages, iteration_count,
    sources_queried) and inspects the message trace to extract both the
    final answer and the tool-retrieved context.
    """
    from langchain_core.messages import HumanMessage

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration_count": 0,
        "sources_queried": [],
    }
    result = agent.invoke(initial_state)
    messages = result["messages"]

    final_answer = messages[-1].content
    retrieved_context = _extract_retrieved_context(messages)

    return final_answer, retrieved_context


def evaluate_single(
    agent,
    test_case: dict,
    judge_llm=None,
) -> EvalResult:
    """Run one test case: query agent, measure latency, judge the answer.

    Args:
        agent: Compiled LangGraph agent.
        test_case: Dict with question_id, category, question, expected_answer,
                   evaluation_criteria.
        judge_llm: LLM for judging. Reuse across calls for efficiency.

    Returns:
        EvalResult with all scores populated.
    """
    question = test_case["question"]
    result = EvalResult(
        question_id=test_case["question_id"],
        category=test_case["category"],
        question=question,
        reference_answer=test_case["expected_answer"],
        model_answer="",
    )

    # Query the agent and measure latency
    t0 = time.monotonic()
    try:
        answer, context = _query_agent_with_context(agent, question)
        result.model_answer = answer
        result.retrieved_context = context
    except Exception as e:
        result.model_answer = f"[ERROR] {e}"
        result.latency_seconds = time.monotonic() - t0
        result.judge_error = f"Agent query failed: {e}"
        return result
    result.latency_seconds = time.monotonic() - t0

    # Judge the answer
    try:
        scores = judge_answer(
            question=question,
            reference_answer=test_case["expected_answer"],
            evaluation_criteria=test_case["evaluation_criteria"],
            retrieved_context=result.retrieved_context,
            model_answer=result.model_answer,
            judge_llm=judge_llm,
        )
        for dim_name in DIMENSIONS:
            dim_data = scores.get(dim_name, {})
            setattr(result, dim_name, DimensionScore(
                score=int(dim_data.get("score", 0)),
                rationale=dim_data.get("rationale", ""),
            ))
    except Exception as e:
        logger.error("Judge failed for Q%d: %s", test_case["question_id"], e)
        result.judge_error = str(e)

    return result


def run_evaluation(
    agent,
    csv_path: Optional[str] = None,
    judge_llm=None,
) -> list[EvalResult]:
    """Run the full evaluation suite.

    Args:
        agent: Compiled LangGraph agent.
        csv_path: Path to test CSV. Defaults to tests/test_queries.csv.
        judge_llm: LLM for judging. Created once and reused if None.

    Returns:
        List of EvalResult, one per test case.
    """
    if judge_llm is None:
        judge_llm = _get_judge_llm()

    test_cases = load_test_cases(csv_path)
    results = []

    logger.info("Starting evaluation of %d test cases...", len(test_cases))
    for i, tc in enumerate(test_cases, 1):
        logger.info("[%d/%d] Q%d: %s", i, len(test_cases), tc["question_id"], tc["category"])
        result = evaluate_single(agent, tc, judge_llm=judge_llm)
        results.append(result)
        logger.info(
            "  -> Score: %d/%d | Latency: %.2fs",
            result.total_score, result.max_possible, result.latency_seconds,
        )

    return results


def summarize_results(results: list[EvalResult]) -> dict:
    """Compute aggregate statistics from evaluation results.

    Returns:
        Dict with per-dimension averages, overall average, latency stats,
        and per-category breakdowns.
    """
    if not results:
        return {}

    n = len(results)

    # Per-dimension averages
    dim_avgs = {}
    for dim in DIMENSIONS:
        scores = [getattr(r, dim).score for r in results if getattr(r, dim) is not None]
        dim_avgs[dim] = round(sum(scores) / len(scores), 2) if scores else 0.0

    # Overall
    total_scores = [r.total_score for r in results]
    latencies = [r.latency_seconds for r in results]

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"scores": [], "latencies": []}
        categories[cat]["scores"].append(r.total_score)
        categories[cat]["latencies"].append(r.latency_seconds)

    cat_summary = {}
    for cat, data in categories.items():
        cat_summary[cat] = {
            "count": len(data["scores"]),
            "avg_score": round(sum(data["scores"]) / len(data["scores"]), 2),
            "avg_latency": round(sum(data["latencies"]) / len(data["latencies"]), 2),
        }

    return {
        "total_questions": n,
        "dimension_averages": dim_avgs,
        "overall_avg_score": round(sum(total_scores) / n, 2),
        "overall_max_score": len(DIMENSIONS) * 5,
        "latency_avg": round(sum(latencies) / n, 2),
        "latency_max": round(max(latencies), 2),
        "latency_min": round(min(latencies), 2),
        "category_breakdown": cat_summary,
    }
