"""
hybrid_search.py

负责：
1. BM25 索引缓存
2. 检索策略决策
3. 混合向量 + BM25 检索
4. Rerank 精排
"""

import jieba
import requests
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# ───────── BM25 缓存 ─────────
def get_bm25_index(vector_store, st):
    """
    构建或复用 BM25 模型
    """
    all_content = vector_store.get()
    if not all_content or not all_content["documents"]:
        return None, [], []

    raw_texts = all_content["documents"]
    metadatas = all_content["metadatas"]
    doc_count = len(raw_texts)

    cached = st.session_state.get("bm25_cache")
    if cached and cached["doc_count"] == doc_count:
        return cached["model"], raw_texts, metadatas

    tokenized_corpus = [list(jieba.cut(doc)) for doc in raw_texts]
    bm25_model = BM25Okapi(tokenized_corpus)
    st.session_state["bm25_cache"] = {"model": bm25_model, "doc_count": doc_count}

    return bm25_model, raw_texts, metadatas

# ───────── 检索策略 ─────────
def decide_retrieval_strategy(query: str, llm_client) -> str:
    """
    判断问题属于哪种检索策略：
        skip / keyword / vector / hybrid
    """
    prompt = (
        "判断以下用户问题需要哪种RAG检索策略，只输出一个词：\n"
        "- skip：纯闲聊/问候\n"
        "- keyword：含精确食材名或菜名\n"
        "- vector：语义型问题\n"
        "- hybrid：复杂综合问题（食材替换/饮食限制/多食材搭配）\n\n"
        f"用户问题：{query}\n"
        "只输出：skip / keyword / vector / hybrid"
    )
    try:
        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        strategy = resp.choices[0].message.content.strip().lower()
        return strategy if strategy in ("skip", "keyword", "vector", "hybrid") else "hybrid"
    except Exception:
        return "hybrid"

# ───────── 混合检索 ─────────
def execute_hybrid_search(
    query: str,
    vector_store,
    st,
    k_vector: int = 4,
    k_bm25: int = 4
) -> list:
    """
    BM25 + 向量双路检索
    """
    # 向量检索
    all_vector = vector_store.similarity_search(query, k=k_vector * 2)
    vector_results = [
        d for d in all_vector
        if d.page_content not in ("Database initialized.", "Database initialized successfully.")
        and d.metadata.get("chunk_type") != "large"
    ][:k_vector]

    # BM25 通路
    bm25_results = []
    try:
        bm25_model, raw_texts, metadatas = get_bm25_index(vector_store, st)
        if bm25_model:
            tokenized_query = list(jieba.cut(query))
            scores = bm25_model.get_scores(tokenized_query)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_bm25]
            bm25_results = [
                Document(page_content=raw_texts[i], metadata=metadatas[i])
                for i in top_idx
                if scores[i] > 0
                and metadatas[i].get("chunk_type") != "large"
                and raw_texts[i] not in ("Database initialized.", "Database initialized successfully.")
            ]
    except Exception:
        pass

    # 去重
    seen, combined = set(), []
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    return combined

# ───────── Rerank 精排 ─────────
def execute_rerank(
    query: str,
    docs: list,
    api_key: str,
    rerank_url: str,
    top_n: int = 3
) -> list:
    """
    调用 Rerank 服务进行精排
    """
    if not docs:
        return []

    try:
        headers  = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.post(
            rerank_url,
            headers=headers,
            json={
                "model": "rerank-3",
                "query": query,
                "documents": [d.page_content for d in docs],
                "top_n": top_n
            },
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        reranked = []
        for item in results:
            doc = docs[item["index"]]
            doc.metadata["rerank_score"] = round(item["relevance_score"], 4)
            reranked.append(doc)
        return reranked
    except Exception:
        return docs[:top_n]