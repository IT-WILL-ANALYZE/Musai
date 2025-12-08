import streamlit as st
import datetime
import textwrap
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple

# === Local Import === 
import llm.models as models
import embed.langchain_chroma_embed as embed
import prompt.langchain_hub as prpt

# === LLM (OpenAI or Claude) ===
from openai import OpenAI
llm = models.set_model()


# -----------------------------------------------------------------------------
# Streamlit settings
# -----------------------------------------------------------------------------

st.set_page_config(page_title="MusAi - 패션분석에 대한 모든것", page_icon="✨")

HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
CONTEXT_LEN = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=2)

executor = ThreadPoolExecutor(max_workers=5)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def build_prompt(**kwargs):
    parts = []
    for key, value in kwargs.items():
        if value:
            parts.append(f"<{key}>\n{value}\n</{key}>")
    return "\n".join(parts)


def history_to_text(chat_history):
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


def generate_chat_summary(messages):
    prompt = build_prompt(
        instructions="Summarize this conversation concisely.",
        conversation=history_to_text(messages),
    )
    return llm.invoke(prompt)


# -----------------------------------------------------------------------------
# Main RAG prompt builder
# -----------------------------------------------------------------------------

def build_question_prompt(question):
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]

    recent_history_str = history_to_text(recent_history)

    task_infos = []
    TaskInfo = namedtuple("TaskInfo", ["name", "func", "args"])

    # 오래된 기록 요약
    if SUMMARIZE_OLD_HISTORY and old_history:
        task_infos.append(TaskInfo("old_summary", generate_chat_summary, (old_history,)))

    # 문서 검색
    chroma_search = embed.get_similary_vector(question)
    task_infos.append(TaskInfo("rag_context", chroma_search, (question,)))

    # 병렬 실행
    results = executor.map(lambda t: (t.name, t.func(*t.args)), task_infos)
    context = {name: result for name, result in results}

    prompt = build_prompt(
        instructions=textwrap.dedent(prpt.rag_prompt),
        old_summary=context.get("old_summary"),
        rag_context=context.get("rag_context"),
        recent_messages=recent_history_str,
        question=question,
    )
    return prompt


# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------

st.title("MusAi - 무신사 데이터 기반, 패션관련 모든것에 대해 답해드립니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_message = st.chat_input("질문을 입력해주세요")

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Main interaction
if user_message:
    # Show user msg
    with st.chat_message("user"):
        st.markdown(user_message)

    st.session_state.messages.append({"role": "user", "content": user_message})

    # Build prompt
    with st.spinner("문서 검색 및 프롬프트 구성 중..."):
        full_prompt = build_question_prompt(user_message)

    # LLM response streaming
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = st.write_stream(llm.invoke({"question":full_prompt},stream=True))

    st.session_state.messages.append({"role": "assistant", "content": response})
