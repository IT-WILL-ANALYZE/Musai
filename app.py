from config import setting
import streamlit as st
import copy
import uuid
import os as os
import rag_chain  as rag_chain
from langchain_core.documents import Document


# ------------------------------
# 전역변수/세션관리
# ------------------------------
ADMIN_CODE = os.getenv("ADMIN_CODE")  # .env에서 읽기, 없으면 기본값
save_dir = "rag_resources/uploads"

if "admin_step_up" not in st.session_state:
    st.session_state.admin_step_up = 0 # 0 : 기본  / 1 : 청킹 / 2 : 임베딩 / 3: 테스트
if "admin_verified" not in st.session_state:
    st.session_state.admin_verified = True # 관리자 인증 여부 - 기본값 False
if "test_mode" not in st.session_state:
    st.session_state.test_mode = False
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""
if "chunked_docs" not in st.session_state:
    st.session_state.chunked_docs = []
if "original_chunked_docs" not in st.session_state:
    st.session_state.original_chunked_docs = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = ""


# ------------------------------
# 관리자 - rag 생성
# ------------------------------
def create_vectordb(file_url):
    with st.spinner("임베딩 진행 중..."):
        success, vectordb = rag_chain.get_vectordb(st.session_state.chunked_docs)
        if success:
            st.session_state.vectordb = vectordb
            st.success("벡터DB 생성 성공!")
        else:
            st.error("벡터DB 생성 실패!")


def admin_chunk_rag(file_url):
    if st.session_state.admin_verified and st.session_state.test_mode:
        with st.spinner("청킹 진행 중..."):
            chunks = rag_chain.get_chunked_docs(file_url)

        st.session_state.admin_step_up = 1      # 임베딩 버튼 생성
        st.session_state.chunked_docs = chunks  # chunk 세션 관리
        st.session_state.original_chunked_docs = copy.deepcopy(chunks)

        render_chunk_manager()
        st.rerun()


def reset_chunks():
    if "original_chunked_docs" not in st.session_state:
        st.warning("원본 Chunk 정보가 없습니다.")
        return

    # 1️⃣ 원본을 deep copy
    restored_docs = copy.deepcopy(
        st.session_state.original_chunked_docs
    )

    # 2️⃣ UUID 재발급 (⭐ 매우 중요)
    for doc in restored_docs:
        doc.metadata["id"] = str(uuid.uuid4())

    # 3️⃣ 현재 chunk 교체
    st.session_state.chunked_docs = restored_docs

    # 4️⃣ text_area 관련 state 전부 제거
    keys_to_delete = [
        key for key in st.session_state.keys()
        if key.startswith("chunk_edit_")
    ]

    for key in keys_to_delete:
        del st.session_state[key]


def save_vectordb():
    if st.session_state.admin_verified and st.session_state.test_mode and st.session_state.vectordb :
        with st.spinner("데이터베이스 저장 중..."):
            success = rag_chain.set_vectordb(os.path.join(save_dir, st.session_state.uploaded_filename), st.session_state.chunked_docs)
            if success:
                st.success("저장 성공!")
            else:
                st.error("저장 실패!")

# -------------------------------------------------
# Chunk Manager UI
# -------------------------------------------------
def render_chunk_manager():
    if "chunked_docs" not in st.session_state:
        st.session_state.chunked_docs = []

    if not st.session_state.chunked_docs:
        st.info("청킹된 문서가 없습니다.")
        return

    st.subheader("🧩 Chunk 관리자")

    delete_target_idx = None
    add_after_idx = None

    for idx, doc in enumerate(st.session_state.chunked_docs):
        chunk_id = doc.metadata.get("id")

        # 안전장치: id 없으면 생성
        if not chunk_id:
            chunk_id = str(uuid.uuid4())
            doc.metadata["id"] = chunk_id

        with st.expander(f"Chunk {idx + 1}", expanded=False):
            st.markdown(f"**관련 항목:** `{doc.metadata.get('title', 'N/A')}`")
            st.markdown(f"**카테고리:** `{doc.metadata.get('category', 'N/A')}`")

            edited_text = st.text_area(
                "Chunk Content",
                value=doc.page_content,
                height=200,
                key=f"chunk_edit_{chunk_id}"
            )

            # ✏️ 내용 수정 반영
            if edited_text != doc.page_content:
                doc.page_content = edited_text

            # 버튼 영역
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("➕ 추가", key=f"add_{chunk_id}", type="secondary", use_container_width=True):
                    add_after_idx = idx
            
            with btn_col2:
                if st.button("🗑 삭제", key=f"delete_{chunk_id}", type="secondary", use_container_width=True):
                    delete_target_idx = idx

    # -----------------------------
    # 삭제 처리 (루프 밖에서!)
    # -----------------------------
    if delete_target_idx is not None:
        st.session_state.chunked_docs.pop(delete_target_idx)
        st.success(f"✅ Chunk {delete_target_idx + 1} 삭제 완료!")
        st.rerun()

    # -----------------------------
    # 추가 처리
    # -----------------------------
    if add_after_idx is not None:
        st.session_state.add_after_chunk_idx = add_after_idx
        st.rerun()

    # -------------------------------------------------
    # Chunk 추가 폼 (특정 위치 다음에)
    # -------------------------------------------------
    if "add_after_chunk_idx" in st.session_state:
        add_idx = st.session_state.add_after_chunk_idx
        
        st.divider()
        st.subheader(f"➕ Chunk {add_idx + 1} 다음에 추가")
        
        with st.form("add_chunk_at_index_form", clear_on_submit=True):
            new_chunk_text = st.text_area(
                "새 Chunk 내용",
                height=150,
                placeholder="여기에 새로운 chunk 내용을 입력하세요..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                new_title = st.text_input("관련 항목", value="Manual")
            with col2:
                new_category = st.text_input("카테고리", value="Manual")
            
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                submitted = st.form_submit_button("➕ 추가", type="primary", use_container_width=True)
            with form_col2:
                cancelled = st.form_submit_button("❌ 취소", use_container_width=True)
            
            if submitted and new_chunk_text.strip():
                new_doc = Document(
                    page_content=new_chunk_text.strip(),
                    metadata={
                        "id": str(uuid.uuid4()),
                        "source": "manual",
                        "title": new_title,
                        "category": new_category
                    }
                )
                
                # 지정된 위치 다음에 삽입
                st.session_state.chunked_docs.insert(add_idx + 1, new_doc)
                st.success(f"✅ Chunk {add_idx + 1} 다음에 추가 완료!")
                del st.session_state.add_after_chunk_idx
                st.rerun()
            
            if cancelled:
                del st.session_state.add_after_chunk_idx
                st.rerun()

    # -------------------------------------------------
    # 원본 복구
    # -------------------------------------------------
    st.divider()
    if st.button("↩ 원본 Chunk로 되돌리기"):
        reset_chunks()
        st.rerun()


# ------------------------------
# 다이어로그(Modal)
# ------------------------------
@st.dialog("MusAi란")
def show_who_am_i():
    st.caption("""
        Musai란 무신사의 데이터를 기반으로 최신 트렌드 및...\n
        어쩌구 저쩌구\n
        ...
    """)
    
@st.dialog("관리자모드")
def show_upload_file():

    # 1) 관리자 인증 단계
    if not st.session_state.admin_verified:
        with st.form("admin_auth_form"):
            admin_input = st.text_input("관리자 코드를 입력하세요", type="password")
            submit = st.form_submit_button("확인")

        if submit:
            if admin_input == ADMIN_CODE:
                st.session_state.admin_verified = True
                st.success("인증 성공! 파일을 업로드하세요.")
                st.rerun()
            else:
                st.error("잘못된 코드입니다.")
        return

    # 2) 업로드 UI
    uploader_key = f"uploader_{st.session_state.uploader_version}"

    uploaded = st.file_uploader(
        "학습 데이터 업로드",
        type=["md", "markdown", "txt", "pdf", "docx", "csv", "xlsx", "xls"],
        key=uploader_key
    )

    # 업로드된 경우 이름만 기록
    if uploaded:
        save_path = os.path.join(save_dir, uploaded.name)
        st.session_state.uploaded_filename = uploaded.name
        st.session_state.test_mode = False

        st.success(f"파일 선택됨: {uploaded.name}")

        if os.path.exists(save_path):
            st.error(f"⚠ '{uploaded.name}' 파일은 이미 존재합니다. 테스트 진행시 덮어씌워집니다.")

        # -------------------------
        # 업로드 버튼
        # -------------------------
        if st.button("업로드", key="btn_upload", type="primary"):
            st.session_state.admin_step_up = 0
            st.session_state.test_mode = True
            st.session_state.uploader_version += 1

            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.rerun()

            
# ------------------------------
# 예시 질문
# ------------------------------
SUGGESTIONS = {
    ":blue[:material/local_library:] MusAi가 뭔가요?": "MusAi가 뭔가요?",
    ":green[:material/local_library:] 최신 트렌드에 대해 조사해줘": "최신 트렌드에 대해 조사해줘",
    ":red[:material/local_library:] (무신사 상품번호)에 대해 조사해줘": "(무신사 상품번호)에 대해 조사해줘",
}


# ------------------------------
# 기본 UI 설정
# ------------------------------
st.set_page_config(
    page_title="MusAi - 패션분석에 대한 모든 것",
    page_icon="/images/logo.png",
)

st.image("images/logo.png", width=120)

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

# ------------------------------
# 세션 상태 초기화
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initial_question" not in st.session_state:
    st.session_state.initial_question = None

if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = None


# ------------------------------
# 상단 타이틀 + Restart 버튼
# ------------------------------
def clear_conversation():
    st.session_state.messages = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None

with title_row:
    # title_row 안에서
    st.title("MusAi (Admin)" if st.session_state.admin_verified else "MusAi", anchor=False, width="stretch")


    # 채팅이 시작된 이후에만 Restart 보여주기
    if len(st.session_state.messages) > 0 or \
       st.session_state.initial_question or \
       st.session_state.selected_suggestion:
        
        st.button(
            "Restart",
            icon=":material/refresh:",
            on_click=clear_conversation,
        )


# ------------------------------
# 상태 체크 (초기 진입 여부)
# ------------------------------
user_just_asked_initial_question = bool(st.session_state.initial_question)
user_just_clicked_suggestion = bool(st.session_state.selected_suggestion)

user_first_interaction = (
    user_just_asked_initial_question or user_just_clicked_suggestion
)

has_message_history = len(st.session_state.messages) > 0


# ==========================================================
# 1) 아직 아무 상호작용도 없을 때: 초기 화면
# ==========================================================
if not user_first_interaction and not has_message_history:
    with st.container():
        # 첫 질문 입력 (값은 st.session_state.initial_question 에 저장됨)
        st.chat_input("질문해주세요.", key="initial_question")

        # 예시 질문 선택
        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    # 하단 버튼 (좌 / 우 배치)
    left, spacer, right = st.columns([1, 6, 1.2])

    with left:
        st.button(
            "&nbsp;:small[:gray[:material/Cognition: MusAi란]]",
            type="tertiary",
            on_click=show_who_am_i,
            use_container_width=True,
        )

    with right:
        st.button(
            "&nbsp;:small[:gray[:material/Face: 관리자 모드]]"
            if not st.session_state.admin_verified
            else "&nbsp;:small[:gray[:material/upload: 파일업로드]]",
            type="tertiary",
            on_click=show_upload_file,
            use_container_width=True,
        )

    # ==============================================
    # 테스트 모드 후: 본문에 청킹/임베딩 버튼 표시
    # ==============================================
    if st.session_state.test_mode and st.session_state.uploaded_filename:
        st.divider()
        st.subheader("📂 파일 테스트 준비 완료")

        st.info(f"선택된 파일: {st.session_state.uploaded_filename}")

        file_url = save_dir + "/" + st.session_state.uploaded_filename
        
        if st.session_state.admin_step_up == 0 :
            if st.button("🧩 청킹 실행"):
                admin_chunk_rag(file_url)
        elif st.session_state.admin_step_up == 1 :
            st.success("ETL 완료! 아래에서 Chunk를 관리할 수 있습니다.")
            if st.button("🧬 임베딩 실행"):
                create_vectordb(file_url)
            render_chunk_manager()

    # 아직 채팅 히스토리가 없으므로 여기서 종료
    st.stop()


# ==========================================================
# 2) 채팅 화면: 기존 메시지 표시
# ==========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ==========================================================
# 3) 입력창 (초기 + 후속 질문 모두 여기서 처리)
# ==========================================================
user_message = st.chat_input("질문해주세요.")
if st.session_state.admin_verified and st.session_state.test_mode and st.session_state.vectordb :
    st.caption(
        f"&nbsp;:small[:gray[:material/file_open: 적용된 파일]] : {st.session_state.uploaded_filename}"    
    )
    
    if st.button("💾 vectorDB에 저장"):
        save_vectordb()
        

# chat_input 에서 바로 입력이 안 들어온 경우
# (ex. 첫 진입 시 initial_question / suggestion 으로 들어온 경우)
if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    elif user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]


# 실제로 처리할 메시지가 존재하는 경우
if user_message:
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None

    # 유저 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_message})

    # --- AI 응답 생성 ---
    if st.session_state.admin_verified and st.session_state.test_mode and st.session_state.vectordb:
        with st.spinner("🤔 MusAi가 답변을 생성하고 있어요..."):
            ai_response = st.write_stream(rag_chain.get_llm_response_temp(
                st.session_state.vectordb, 
                user_message,
                history=st.session_state.messages
            ))
    else:
        with st.spinner("🤔 MusAi가 답변을 생성하고 있어요..."):
            ai_response = st.write_stream(rag_chain.get_llm_response(user_message,history=st.session_state.messages))

    # --- assistant 메시지 저장 ---
    st.session_state.messages.append(
        {"role": "assistant", "content": ai_response}
    )

    st.rerun()