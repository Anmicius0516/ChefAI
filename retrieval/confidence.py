"""
confidence.py

负责：
1. Retrieval Confidence
2. Refusal Decision
3. Iterative Retrieval
"""

from typing import List, Tuple


FAITHFULNESS_THRESHOLD = 0.25
SUPPLEMENT_THRESHOLD = 0.15


def check_retrieval_confidence(
    docs: List,
    threshold: float = FAITHFULNESS_THRESHOLD
) -> Tuple[bool, float]:
    """
    返回:
        (是否可信, 最大分数)
    """

    if not docs:
        return False, 0.0

    scores = [
        doc.metadata["rerank_score"]
        for doc in docs
        if "rerank_score" in doc.metadata
    ]

    if not scores:
        return True, 1.0

    max_score = max(scores)

    return (
        max_score >= threshold,
        max_score
    )


def need_supplement_retrieval(
    score: float,
    threshold: float = FAITHFULNESS_THRESHOLD,
    supplement_threshold: float = SUPPLEMENT_THRESHOLD,
) -> bool:
    """
    判断是否需要补充检索

    score >= threshold
        已可信

    score < supplement_threshold
        直接拒答

    supplement_threshold <= score < threshold
        补充检索
    """

    return (
        supplement_threshold
        <= score
        < threshold
    )


def execute_iterative_retrieval(
    query: str,
    initial_docs: list,
    llm_client,
    hybrid_search_fn,
    rerank_fn,
):
    """
    边界置信度触发补充检索

    参数:
        llm_client
            OpenAI/Zhipu Client

        hybrid_search_fn
            execute_hybrid_search

        rerank_fn
            execute_rerank
    """

    is_confident, score = check_retrieval_confidence(
        initial_docs
    )

    if is_confident:
        return initial_docs

    if score < SUPPLEMENT_THRESHOLD:
        return initial_docs

    try:

        prompt = f"""
原始问题：

{query}

请从不同角度生成2个更精确的食谱检索子问题。

要求：
1. 使用同义食材
2. 使用相关烹饪技法
3. 每行一个
4. 不要编号
"""

        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100
        )

        sub_queries = [
            q.strip()
            for q in resp.choices[0]
            .message.content
            .split("\n")
            if q.strip()
        ][:2]

    except Exception:
        return initial_docs

    all_docs = list(initial_docs)

    seen = {
        doc.page_content
        for doc in initial_docs
    }

    for sub_query in sub_queries:

        extra_docs = hybrid_search_fn(
            sub_query,
            k_vector=3,
            k_bm25=3
        )

        for doc in extra_docs:

            if doc.page_content not in seen:

                all_docs.append(doc)

                seen.add(
                    doc.page_content
                )

    return rerank_fn(
        query,
        all_docs,
        top_n=5
    )