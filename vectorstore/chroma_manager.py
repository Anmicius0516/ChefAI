"""
chroma_manager.py

负责：
1. Embedding Service
2. Chroma初始化
3. 文档入库
4. 数据源管理
"""

import os

import streamlit as st

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from retrieval.parent_retriever import (
    split_documents_hierarchical,
)


# =====================================================
# Embedding Service
# =====================================================

class ZhipuEmbeddingService:

    def __init__(
        self,
        client,
        model_name="embedding-3"
    ):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts):

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        return [
            item.embedding
            for item in response.data
        ]

    def embed_query(self, text):

        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )

        return response.data[0].embedding


# =====================================================
# Chroma初始化
# =====================================================

@st.cache_resource
def init_vector_store(db_path: str, _embedding_service):
    # _embedding_service 会被 Streamlit 忽略 hash
    return Chroma(
        persist_directory=db_path,
        embedding_function=_embedding_service
    )
    


# =====================================================
# 文档入库
# =====================================================

def ingest_documents(
    vector_store,
    raw_docs,
    source_name,
):

    _, small_chunks = (
        split_documents_hierarchical(
            raw_docs,
            source_name
        )
    )

    if not small_chunks:
        return 0

    vector_store.add_documents(
        small_chunks
    )

    st.session_state.pop(
        "bm25_cache",
        None
    )

    return len(
        small_chunks
    )


# =====================================================
# 数据源管理
# =====================================================

def get_all_sources(
    vector_store
):

    all_data = vector_store.get()

    sources = sorted(
        set(
            meta.get("source", "")
            for meta in all_data.get(
                "metadatas",
                []
            )
            if meta.get(
                "source",
                ""
            )
            not in (
                "system",
                "",
                "Database initialized.",
                "Database initialized successfully."
            )
        )
    )

    return sources


def get_source_info(
    vector_store,
    source_name: str
):

    all_data = vector_store.get()

    metas = [
        meta
        for meta in all_data.get(
            "metadatas",
            []
        )
        if meta.get(
            "source"
        ) == source_name
    ]

    if not metas:
        return {
            "count": 0,
            "file_type": "unknown"
        }

    return {
        "count": len(metas),
        "file_type":
        metas[0].get(
            "file_type",
            "unknown"
        )
    }


# =====================================================
# 删除数据源
# =====================================================

def delete_source_from_db(
    vector_store,
    source_name: str
):

    all_data = vector_store.get()

    ids_to_delete = [

        doc_id

        for doc_id, meta in zip(
            all_data["ids"],
            all_data["metadatas"]
        )

        if meta.get(
            "source"
        ) == source_name
    ]

    if ids_to_delete:

        vector_store.delete(
            ids=ids_to_delete
        )

        st.session_state.pop(
            "bm25_cache",
            None
        )

    return len(
        ids_to_delete
    )