"""
evaluator.py

负责：

1. Retrieval Quality
2. Faithfulness
3. Answer Relevance
4. Context Precision

用于:
- 调试RAG
- Benchmark
- Demo展示
"""

from typing import List, Dict


# =====================================================
# Retrieval Evaluation
# =====================================================

def evaluate_retrieval(
    retrieved_docs: List
) -> Dict:
    """
    检查检索质量
    """

    if not retrieved_docs:

        return {
            "retrieval_score": 0.0,
            "doc_count": 0
        }

    scores = [

        doc.metadata.get(
            "rerank_score",
            0.0
        )

        for doc in retrieved_docs
    ]

    retrieval_score = (
        sum(scores)
        / len(scores)
    )

    return {
        "retrieval_score":
        round(retrieval_score, 4),

        "doc_count":
        len(retrieved_docs)
    }


# =====================================================
# Faithfulness
# =====================================================

def evaluate_faithfulness(
    answer: str,
    contexts: List
) -> float:
    """
    简化Faithfulness

    后续可替换为:
    Ragas
    DeepEval
    LLM Judge
    """

    if not contexts:
        return 0.0

    context_text = "\n".join(
        [
            doc.page_content[:500]
            for doc in contexts
        ]
    )

    overlap = 0

    answer_tokens = set(
        answer.split()
    )

    context_tokens = set(
        context_text.split()
    )

    overlap = len(
        answer_tokens
        &
        context_tokens
    )

    if not answer_tokens:
        return 0.0

    return round(
        overlap / len(answer_tokens),
        4
    )


# =====================================================
# Context Precision
# =====================================================

def evaluate_context_precision(
    retrieved_docs: List
) -> float:
    """
    检索上下文质量
    """

    if not retrieved_docs:
        return 0.0

    scores = [

        doc.metadata.get(
            "rerank_score",
            0
        )

        for doc in retrieved_docs
    ]

    return round(
        sum(scores)
        / len(scores),
        4
    )


# =====================================================
# Answer Relevance
# =====================================================

def evaluate_answer_relevance(
    question: str,
    answer: str
) -> float:
    """
    简单版相关性评估
    """

    q_words = set(
        question.split()
    )

    a_words = set(
        answer.split()
    )

    if not q_words:
        return 0.0

    overlap = len(
        q_words & a_words
    )

    return round(
        overlap / len(q_words),
        4
    )


# =====================================================
# Unified Evaluation
# =====================================================

def evaluate_all(
    question: str,
    answer: str,
    contexts: List
):

    retrieval = evaluate_retrieval(
        contexts
    )

    faithfulness = (
        evaluate_faithfulness(
            answer,
            contexts
        )
    )

    context_precision = (
        evaluate_context_precision(
            contexts
        )
    )

    relevance = (
        evaluate_answer_relevance(
            question,
            answer
        )
    )

    return {

        "retrieval_score":
        retrieval["retrieval_score"],

        "retrieved_docs":
        retrieval["doc_count"],

        "faithfulness":
        faithfulness,

        "context_precision":
        context_precision,

        "answer_relevance":
        relevance,
    }