# app.py

"""
ChefAI 烹饪与食谱助手 · 精简版
依赖拆分模块：
- retrieval/
- memory/
- parsers/
- vectorstore/
- evaluation/
"""

import os
import gc
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from vectorstore.chroma_manager import (
    ZhipuEmbeddingService,
    init_vector_store,
    ingest_documents,
    get_all_sources,
    get_source_info,
    delete_source_from_db
)

from retrieval.hybrid_search import (
    get_bm25_index,
    execute_hybrid_search,
    execute_rerank,
    decide_retrieval_strategy
)

from retrieval.parent_retriever import split_documents_hierarchical
from retrieval.confidence import (
    FAITHFULNESS_THRESHOLD,
    check_retrieval_confidence,
    execute_iterative_retrieval
)
from memory.conversation_memory import (
    get_effective_history,
    execute_query_rewriting
)

from parsers.pdf_parser import parse_pdf
from parsers.image_parser import parse_image
from parsers.excel_parser import parse_excel_csv
from parsers.url_parser import parse_url

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
load_dotenv()

API_KEY    = os.getenv("ZHIPU_API_KEY")
BASE_URL   = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
RERANK_URL = os.getenv("ZHIPU_RERANK_URL", "https://open.bigmodel.cn/api/paas/v4/tools/rerank")
DB_PATH    = os.getenv("CHROMA_DB_PATH", "./chroma_db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./temp_upload")

if not API_KEY:
    st.set_page_config(page_title="配置错误", page_icon="❌")
    st.error("未检测到 `ZHIPU_API_KEY` 环境变量")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
embedding_service = ZhipuEmbeddingService(client)
vector_store = init_vector_store(DB_PATH, embedding_service)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="ChefAI · 烹饪助手", layout="wide")
st.title("🍳 ChefAI · 烹饪与食谱助手")

with st.sidebar:
    st.header("🍽️ 食谱库管理")
    st.markdown("上传食谱文件或输入美食博客 URL 注入数据")

    uploaded_file = st.file_uploader(
        "上传食谱文件",
        type=["txt", "pdf", "docx", "md", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "bmp"]
    )

    if uploaded_file is not None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        inject_trigger = st.button("▶ 开始注入", use_container_width=True)
        if inject_trigger:
            ext = os.path.splitext(uploaded_file.name)[-1].lower()
            raw_docs = []
            try:
                if ext == ".pdf":
                    raw_docs = parse_pdf(file_path, uploaded_file.name, client)
                elif ext in [".xlsx", ".xls", ".csv"]:
                    raw_docs = parse_excel_csv(file_path, uploaded_file.name)
                elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                    raw_docs = parse_image(file_path, uploaded_file.name, client)
                else:
                    from langchain_community.document_loaders import Docx2txtLoader, TextLoader
                    if ext in [".docx", ".doc"]:
                        loader = Docx2txtLoader(file_path)
                    else:
                        loader = TextLoader(file_path, encoding="utf-8")
                    raw_docs = loader.load()
                    for d in raw_docs:
                        d.metadata["source"] = uploaded_file.name
                        d.metadata["file_type"] = "document"
            except Exception as e:
                st.error(f"解析失败：{e}")

            if raw_docs:
                count = ingest_documents(vector_store, raw_docs, uploaded_file.name)
                del raw_docs; gc.collect()
                st.success(f"✅ 注入 {count} 个分块")
                try: os.remove(file_path)
                except: pass

    # URL注入
    url_input = st.text_input("输入美食博客/食谱URL")
    if url_input and st.button("🌐 抓取并注入"):
        raw_docs, err = parse_url(url_input)
        if err: st.error(err)
        elif raw_docs:
            count = ingest_documents(vector_store, raw_docs, url_input)
            del raw_docs; gc.collect()
            st.success(f"✅ 博客已注入 {count} 个分块")

    # 数据源管理
    st.divider()
    st.markdown("##### 🗂️ 食谱库管理")
    all_sources = get_all_sources(vector_store)
    if not all_sources:
        st.caption("食谱库暂无数据")
    else:
        for src in all_sources:
            info = get_source_info(vector_store, src)
            col1, col2 = st.columns([3,1])
            col1.markdown(f"**{src}** | {info['count']} 块 · {info['file_type']}")
            if col2.button("🗑️", key=f"del_{src}"):
                deleted = delete_source_from_db(vector_store, src)
                st.toast(f"已删除 {deleted} 块", icon="🗑️")
                st.experimental_rerun()

# ─────────────────────────────────────────────
# 聊天流程
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

user_question = st.chat_input("问我烹饪问题")  
if not user_question: st.stop()

with st.chat_message("user"): st.markdown(user_question)

# 1️⃣ 检索策略
strategy = decide_retrieval_strategy(user_question, client)

# 2️⃣ Query改写
search_query = execute_query_rewriting(user_question, st.session_state.messages, client)

# 3️⃣ 执行检索
candidate_docs = execute_hybrid_search(search_query, vector_store, st, k_vector=4, k_bm25=4)
reranked_docs  = execute_rerank(search_query, candidate_docs, API_KEY, RERANK_URL, top_n=5)

# 4️⃣ 多轮补充检索
final_docs = execute_iterative_retrieval(search_query, reranked_docs, client,
                                         lambda q, k_vector, k_bm25: execute_hybrid_search(q, vector_store, st, k_vector, k_bm25),
                                         lambda q, docs, top_n: execute_rerank(q, docs, API_KEY, RERANK_URL, top_n))

# 5️⃣ 置信度检查
is_confident, max_score = check_retrieval_confidence(final_docs)

# 6️⃣ 上下文 + Memory
recent_history, summary = get_effective_history(st.session_state.messages, client)
summary_section = f"\n【用户偏好记忆】\n{summary}" if summary else ""
context_payload = "\n\n".join([
    f"【食谱参考 [{i+1}]】来源: {d.metadata.get('source','未知')} "
    f"| 菜系: {d.metadata.get('cuisine','—')} "
    f"| 烹饪方式: {d.metadata.get('method','—')} "
    f"| 标签: {d.metadata.get('diet_tags','—')}\n内容: {d.page_content}"
    for i, d in enumerate(final_docs)
])

# 7️⃣ LLM推理
system_prompt = f"""你是 ChefAI，一个专业烹饪助手。
{summary_section}
【回答规范】：
- 引用资料编号[[数字]]
- 标明食材替换方案
- 来源表格/图片/网页标注
【食谱参考资料】:
{context_payload}"""

with st.chat_message("assistant"):

    messages_payload = (
        [{"role":"system","content":system_prompt}]
        + recent_history
        + [{"role":"user","content":user_question}]
    )

    stream = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages_payload,
        stream=True
    )

    raw_output = ""

    placeholder = st.empty()

    for chunk in stream:

        token = chunk.choices[0].delta.content or ""

        raw_output += token

        placeholder.markdown(raw_output)

    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_question
        }
    )

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":raw_output
        }
    )
    st.session_state.messages.append({"role":"user","content":user_question})
    st.session_state.messages.append({"role":"assistant","content":raw_output})