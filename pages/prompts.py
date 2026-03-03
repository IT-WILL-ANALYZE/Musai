import streamlit as st
import os
import re
from loguru import logger
from llm.models import ALLOWED_MODELS
from prompts.load_prompt import get_prompt_txt
import rag_chain

# RAG context (vectordb) - optional
try:
    import embedders.langchain_chroma_embed as chroma_embed
    HAS_VECTORDB = True
except Exception:
    HAS_VECTORDB = False

# ------------------------------
# 설정
# ------------------------------
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

st.set_page_config(
    page_title="OOTData - 프롬프트 관리",
    page_icon="📝",
    layout="wide"
)

# ------------------------------
# 세션 상태 초기화
# ------------------------------
if "admin_verified" not in st.session_state:
    st.session_state.admin_verified = False
if "prompt_delete_target" not in st.session_state:
    st.session_state.prompt_delete_target = None
if "prompt_add_mode" not in st.session_state:
    st.session_state.prompt_add_mode = False

# ------------------------------
# 유틸 함수
# ------------------------------
def get_prompt_files():
    """prompts 폴더의 모든 .txt 파일 목록 반환"""
    try:
        files = [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
        return sorted(files)
    except Exception as e:
        logger.error(f"Failed to get prompt files: {e}")
        return []


def get_prompt_filepath(filename):
    return os.path.join(PROMPTS_DIR, filename)


def save_prompt_file(filename, content):
    """프롬프트 파일 저장"""
    try:
        filepath = get_prompt_filepath(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.success(f"Saved prompt: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save prompt {filename}: {e}")
        st.error(f"저장 실패: {e}")
        return False


def delete_prompt_file(filename):
    """프롬프트 파일 삭제"""
    try:
        filepath = get_prompt_filepath(filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.success(f"Deleted prompt: {filename}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete prompt {filename}: {e}")
        st.error(f"삭제 실패: {e}")
    return False


def extract_template_variables(text):
    """프롬프트 텍스트에서 {variable} 형태의 변수 추출"""
    return list(set(re.findall(r"\{(\w+)\}", text)))


# ------------------------------
# UI 렌더링
# ------------------------------
st.title("📝 프롬프트 관리")
st.caption("LLM 프롬프트를 조회, 수정, 추가/삭제하고 테스트할 수 있습니다.")

# 관리자 인증 체크
if not st.session_state.admin_verified:
    st.error("🚨 관리자 모드에서만 접근 가능합니다.")
    st.stop()

prompt_files = get_prompt_files()

# ------------------------------
# 상단: 모델 & 프롬프트 선택
# ------------------------------
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    selected_model = st.selectbox(
        "🤖 LLM 모델",
        options=ALLOWED_MODELS,
        index=0,
        key="prompt_model_select"
    )

with col2:
    if prompt_files:
        selected_prompt = st.selectbox(
            "📄 프롬프트",
            options=prompt_files,
            key="prompt_file_select"
        )
    else:
        selected_prompt = None
        st.info("등록된 프롬프트가 없습니다. 아래에서 새 프롬프트를 추가하세요.")

with col3:
    if st.button("➕ 새 프롬프트", use_container_width=True):
        st.session_state.prompt_add_mode = True
        st.rerun()

# ------------------------------
# 새 프롬프트 추가 폼
# ------------------------------
if st.session_state.prompt_add_mode:
    st.divider()
    st.subheader("➕ 새 프롬프트 추가")
    with st.form("add_prompt_form", clear_on_submit=True):
        new_name = st.text_input("파일명 (.txt)", placeholder="예: my_prompt.txt")
        new_content = st.text_area(
            "프롬프트 내용",
            height=200,
            placeholder="예:\nhuman\n당신은 {role}입니다.\n질문: {question}\n답변:"
        )
        add_col1, add_col2 = st.columns(2)
        with add_col1:
            submitted = st.form_submit_button("추가")
        with add_col2:
            cancelled = st.form_submit_button("취소")

        if submitted and new_name.strip():
            fname = new_name.strip()
            if not fname.endswith(".txt"):
                fname += ".txt"
            if fname in prompt_files:
                st.error(f"'{fname}' 파일이 이미 존재합니다.")
            elif new_content.strip():
                if save_prompt_file(fname, new_content.strip()):
                    st.success(f"✅ {fname} 추가 완료!")
                    st.session_state.prompt_add_mode = False
                    st.rerun()
            else:
                st.warning("프롬프트 내용을 입력해주세요.")

        if cancelled:
            st.session_state.prompt_add_mode = False
            st.rerun()

    st.stop()

# ------------------------------
# 프롬프트 편집 & 삭제 (기존 프롬프트 선택 시)
# ------------------------------
if not selected_prompt:
    st.info("프롬프트를 선택하거나 새로 추가해주세요.")
    st.stop()

st.divider()

# 현재 프롬프트 내용 로드
current_content = get_prompt_txt(selected_prompt)
variables = extract_template_variables(current_content)

# 삭제 확인
if st.session_state.prompt_delete_target == selected_prompt:
    st.warning(f"⚠️ '{selected_prompt}'를 삭제하시겠습니까?")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        if st.button("예, 삭제합니다"):
            if delete_prompt_file(selected_prompt):
                st.session_state.prompt_delete_target = None
                st.success("삭제되었습니다.")
                st.rerun()
    with dcol2:
        if st.button("취소"):
            st.session_state.prompt_delete_target = None
            st.rerun()
    st.stop()

# ------------------------------
# 프롬프트 내용 편집
# ------------------------------
st.subheader(f"📄 {selected_prompt}")

# selectbox 변경 시 해당 프롬프트 내용으로 갱신되도록 key에 selected_prompt 포함
edited_content = st.text_area(
    "프롬프트 내용",
    value=current_content,
    height=300,
    key=f"prompt_edit_area_{selected_prompt}",
    help="question : 질문\n"
        "content  : 분석할 텍스트\n"
        "context  : vectorDB 탐색 내용(vectorDB 선택시 자동 적용)\n"
        "history  : 이전 대화 내용 (테스트 시엔 적용 X)\n\n"
        "템플릿 변수는 {변수명} 형태로 사용합니다.\n"
        "저장 버튼을 눌렀을 때만 .txt 파일로 저장됩니다."
)

btn_col1, btn_col2, _ = st.columns([1, 1, 4])
with btn_col1:
    if st.button("💾 저장", type="primary"):
        if save_prompt_file(selected_prompt, edited_content):
            st.success("저장되었습니다.")
            st.rerun()

with btn_col2:
    if st.button("🗑 삭제", type="secondary"):
        st.session_state.prompt_delete_target = selected_prompt
        st.rerun()

# 저장 후 변수 재추출 (편집본 기준)
variables = extract_template_variables(edited_content)

# ------------------------------
# 프롬프트 테스트
# ------------------------------
st.divider()
st.subheader("🧪 프롬프트 테스트")

if not variables:
    st.info("이 프롬프트에는 템플릿 변수가 없습니다. {변수명} 형태로 추가하면 테스트할 수 있습니다.")
else:
    # history, context는 입력 제외 (history 미사용, context는 VectorDB 체크로만)
    user_vars = [v for v in variables if v not in ("history", "context")]
    use_rag_for_context = False
    if user_vars:
        st.caption(f"입력 변수: {', '.join(user_vars)}")
    if "context" in variables and HAS_VECTORDB:
        use_rag_for_context = st.checkbox("VectorDB에서 검색 (RAG)", value=False, key="use_rag_context")

    # 변수별 입력 필드 (question, content 순 우선, history/context 제외)
    var_order = ["question", "content"]
    ordered_vars = sorted(user_vars, key=lambda v: (var_order.index(v) if v in var_order else 99, v))

    var_inputs = {}

    for var in ordered_vars:
        if var == "question":
            var_inputs[var] = st.text_input(f"{var}", placeholder="질문을 입력하세요", key=f"var_{var}")
        elif var == "content":
            var_inputs[var] = st.text_area(f"{var}", value="", height=150, placeholder="분석할 텍스트를 입력하세요", key=f"var_{var}")
        else:
            var_inputs[var] = st.text_input(f"{var}", value="", key=f"var_{var}")

    # 실행 버튼
    if st.button("▶ 실행", type="primary"):
        missing = [v for v in user_vars if not (var_inputs.get(v) or "").strip()]
        if missing:
            st.warning(f"다음 변수를 입력해주세요: {', '.join(missing)}")
        else:
            query = var_inputs.get("question", var_inputs.get("content", ""))
            st.subheader("결과")
            try:
                with st.spinner("테스트 작업 중..."):
                    st.write_stream(
                        rag_chain.get_llm_response_test(
                            query=query,
                            llm_model=selected_model,
                            prompt_text=edited_content,
                            variables=var_inputs,
                            use_rag_for_context=use_rag_for_context and HAS_VECTORDB,
                        )
                    )
            except Exception as e:
                st.error(f"실행 오류: {e}")
