"""
parent_retriever.py

负责：
1. Parent-Child Chunking
2. Large Chunk管理
3. Small Chunk管理
4. Parent关联
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval.metadata_filter import extract_metadata_tags


def split_documents_hierarchical(
    raw_docs: List[Document],
    source_name: str,
) -> Tuple[List[Document], List[Document]]:
    """
    Parent-Child Chunking

    Large Chunk:
        1500 / 200

    Small Chunk:
        300 / 60

    检索：
        Small Chunk

    生成：
        Parent Chunk
    """

    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60,
    )

    large_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    large_chunks = large_splitter.split_documents(raw_docs)

    small_chunks = []

    for large_idx, large_doc in enumerate(large_chunks):

        parent_id = f"{source_name}_large_{large_idx}"

        large_doc.metadata.update({
            "chunk_id": parent_id,
            "chunk_type": "large",
            "source": source_name,
        })

        child_chunks = small_splitter.split_documents(
            [large_doc]
        )

        for child_doc in child_chunks:

            page = child_doc.metadata.get(
                "page",
                large_doc.metadata.get("page", 0)
            )

            metadata = extract_metadata_tags(
                child_doc.page_content,
                source_name,
                page
            )

            metadata.update({
                "parent_id": parent_id,
                "chunk_type": "small",
            })

            child_doc.metadata.update(metadata)

            small_chunks.append(child_doc)

    return large_chunks, small_chunks


def build_parent_lookup(
    large_chunks: List[Document]
):
    """
    构建 Parent索引
    """

    return {
        doc.metadata["chunk_id"]: doc
        for doc in large_chunks
    }


def retrieve_parent_context(
    retrieved_docs: List[Document],
    parent_lookup: dict
):
    """
    根据命中的small chunk
    回溯对应large chunk
    """

    parent_docs = []

    seen = set()

    for doc in retrieved_docs:

        parent_id = doc.metadata.get("parent_id")

        if not parent_id:
            continue

        if parent_id in seen:
            continue

        parent_doc = parent_lookup.get(parent_id)

        if parent_doc:
            parent_docs.append(parent_doc)
            seen.add(parent_id)

    return parent_docs