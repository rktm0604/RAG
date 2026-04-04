"""
RAG Evaluation Module — Measures retrieval and generation quality.

Implements RAGAS-style metrics:
- Context Precision: How relevant is the retrieved context to the question?
- Context Recall: Does the retrieved context contain the answer?
- Faithfulness: Does the generated answer match the retrieved context?
- Answer Relevance: How relevant is the answer to the question?

Run evaluation:
    python -m rag_eval --question "What is photosynthesis?" --answer "..." --context "..."
"""

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Optional

import ollama

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "llama3.2:3b"


@dataclass
class EvaluationResult:
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevance: float
    overall_score: float
    request_id: str


def _extract_answer_facts(answer: str) -> list[str]:
    """Extract key factual claims from the answer using simple sentence splitting."""
    sentences = re.split(r'[.!?\n]+', answer)
    facts = [s.strip() for s in sentences if len(s.strip()) > 10]
    return facts


def _evaluate_context_precision(question: str, context: str) -> float:
    """Check if retrieved context is relevant to the question."""
    prompt = f"""Rate how relevant the context is to answering the question on a scale of 0-1.
Respond with just a number between 0 and 1 (e.g., 0.85).

Question: {question}

Context:
{context[:2000]}

Relevance score:"""
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        score_text = response["message"]["content"].strip()
        score = float(re.search(r'0?\.\d+|\d+', score_text).group())
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Context precision evaluation failed: {e}")
        return 0.5


def _evaluate_context_recall(ground_truth: str, context: str) -> float:
    """Check if context contains the expected answer information."""
    if not ground_truth.strip():
        return 0.5
    
    prompt = f"""Rate how well the context contains the information needed to answer the question on a scale of 0-1.
Respond with just a number between 0 and 1.

Expected answer information: {ground_truth[:500]}

Context:
{context[:2000]}

Context recall score:"""
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        score_text = response["message"]["content"].strip()
        score = float(re.search(r'0?\.\d+|\d+', score_text).group())
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Context recall evaluation failed: {e}")
        return 0.5


def _evaluate_faithfulness(answer: str, context: str) -> float:
    """Check if the answer is grounded in the retrieved context."""
    facts = _extract_answer_facts(answer)
    if not facts:
        return 0.5
    
    prompt = f"""Rate how well each fact from the answer is supported by the context on a scale of 0-1.
Respond with just a number between 0 and 1.

Answer facts:
{chr(10).join(f"- {f}" for f in facts[:5])}

Context:
{context[:2000]}

Faithfulness score:"""
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        score_text = response["message"]["content"].strip()
        score = float(re.search(r'0?\.\d+|\d+', score_text).group())
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Faithfulness evaluation failed: {e}")
        return 0.5


def _evaluate_answer_relevance(question: str, answer: str) -> float:
    """Check if the answer directly addresses the question."""
    prompt = f"""Rate how relevant the answer is to the question on a scale of 0-1.
Respond with just a number between 0 and 1.

Question: {question}

Answer:
{answer[:1000]}

Answer relevance score:"""
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        score_text = response["message"]["content"].strip()
        score = float(re.search(r'0?\.\d+|\d+', score_text).group())
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Answer relevance evaluation failed: {e}")
        return 0.5


def evaluate_rag(
    question: str,
    answer: str,
    context: str,
    ground_truth: Optional[str] = None,
    model: str = "llama3.2:3b",
) -> EvaluationResult:
    """
    Evaluate a RAG pipeline response.

    Args:
        question: The user's question.
        answer: The generated answer.
        context: The retrieved context chunks.
        ground_truth: Optional reference answer for context recall.
        model: Ollama model to use for evaluation.

    Returns:
        EvaluationResult with individual and overall scores.
    """
    global OLLAMA_MODEL
    OLLAMA_MODEL = model
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Evaluating RAG response")
    
    context_precision = _evaluate_context_precision(question, context)
    context_recall = _evaluate_context_recall(ground_truth or answer, context) if ground_truth else 0.7
    faithfulness = _evaluate_faithfulness(answer, context)
    answer_relevance = _evaluate_answer_relevance(question, answer)
    
    overall = (context_precision + context_recall + faithfulness + answer_relevance) / 4
    
    logger.info(
        f"[{request_id}] Scores - Precision: {context_precision:.2f}, "
        f"Recall: {context_recall:.2f}, Faithfulness: {faithfulness:.2f}, "
        f"Relevance: {answer_relevance:.2f}, Overall: {overall:.2f}"
    )
    
    return EvaluationResult(
        context_precision=context_precision,
        context_recall=context_recall,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        overall_score=overall,
        request_id=request_id,
    )


def evaluate_batch(
    examples: list[dict],
    model: str = "llama3.2:3b",
) -> dict:
    """
    Evaluate multiple RAG examples and return aggregated metrics.

    Args:
        examples: List of dicts with keys: question, answer, context, ground_truth (optional)
        model: Ollama model to use.

    Returns:
        Dict with averaged scores and per-example results.
    """
    results = []
    for ex in examples:
        result = evaluate_rag(
            question=ex["question"],
            answer=ex["answer"],
            context=ex["context"],
            ground_truth=ex.get("ground_truth"),
            model=model,
        )
        results.append(result)
    
    avg_precision = sum(r.context_precision for r in results) / len(results)
    avg_recall = sum(r.context_recall for r in results) / len(results)
    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    avg_relevance = sum(r.answer_relevance for r in results) / len(results)
    avg_overall = sum(r.overall_score for r in results) / len(results)
    
    return {
        "metrics": {
            "avg_context_precision": avg_precision,
            "avg_context_recall": avg_recall,
            "avg_faithfulness": avg_faithfulness,
            "avg_answer_relevance": avg_relevance,
            "avg_overall_score": avg_overall,
        },
        "individual_results": [
            {
                "request_id": r.request_id,
                "question": examples[i]["question"][:50],
                "scores": {
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "faithfulness": r.faithfulness,
                    "answer_relevance": r.answer_relevance,
                    "overall": r.overall_score,
                },
            }
            for i, r in enumerate(results)
        ],
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG responses")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--answer", required=True, help="Generated answer")
    parser.add_argument("--context", required=True, help="Retrieved context")
    parser.add_argument("--ground-truth", help="Reference answer for recall")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model")
    args = parser.parse_args()
    
    result = evaluate_rag(
        question=args.question,
        answer=args.answer,
        context=args.context,
        ground_truth=args.ground_truth,
        model=args.model,
    )
    
    print(f"\n📊 Evaluation Results (ID: {result.request_id})")
    print(f"  Context Precision: {result.context_precision:.2f}")
    print(f"  Context Recall:    {result.context_recall:.2f}")
    print(f"  Faithfulness:      {result.faithfulness:.2f}")
    print(f"  Answer Relevance:  {result.answer_relevance:.2f}")
    print(f"  ─────────────────────")
    print(f"  Overall Score:     {result.overall_score:.2f}")