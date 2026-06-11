"""
conversation_memory.py

负责：
1. 对话历史管理
2. 历史摘要压缩
3. Query改写（多轮指代消解）
"""

from typing import List, Tuple
import streamlit as st

def compress_history_to_summary(messages: List[dict], llm_client, history_compress_rounds: int = 8) -> str:
    """
    当历史超过 history_compress_rounds 轮时，将前段历史压缩为摘要，控制 Token 成本。
    """
    if len(messages) < history_compress_rounds:
        return st.session_state.get("history_summary", "")

    old_messages = messages[:-4]
    history_text = "\n".join(
        [f"{m['role']}: {m['content'][:300]}" for m in old_messages]
    )

    prompt = (
        f"请将以下烹饪对话历史压缩为150字以内的摘要，保留：\n"
        f"1. 用户提过的食材偏好和饮食限制\n"
        f"2. 已推荐过的食谱名称\n"
        f"3. 用户的口味偏好（辣度、清淡等）\n\n"
        f"【对话历史】\n{history_text}\n\n"
        f"请输出摘要："
    )

    try:
        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        summary = resp.choices[0].message.content.strip()
        st.session_state["history_summary"] = summary
        return summary
    except Exception:
        return st.session_state.get("history_summary", "")


def get_effective_history(messages: List[dict], llm_client, history_compress_rounds: int = 8) -> Tuple[List[dict], str]:
    """
    返回 (recent_messages, summary_text)
    recent_messages: 最近4轮对话
    summary_text: 更早历史的摘要（如有）
    """
    recent = messages[-4:] if len(messages) > 4 else messages
    summary = compress_history_to_summary(messages, llm_client, history_compress_rounds)
    return recent, summary


def execute_query_rewriting(user_question: str, chat_history: List[dict], llm_client) -> str:
    """
    多轮指代消解，将最后模糊问题改写为包含完整食材/菜名/烹饪方式的独立查询语句
    """
    if not chat_history:
        return user_question

    history_context = "\n".join(
        [f"{m['role']}: {m['content']}" for m in chat_history[-4:]]
    )

    prompt = (
        f"结合历史对话，将最后的模糊提问改写为包含完整食材/菜名/烹饪方式的独立查询语句。\n"
        f"【历史上下文】\n{history_context}\n"
        f"【当前输入】\n{user_question}\n"
        "请直接输出改写结果，不含任何解释。"
    )

    try:
        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return user_question