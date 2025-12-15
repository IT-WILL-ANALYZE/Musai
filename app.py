import streamlit as st
import os as os
import time
import rag_chain  as rag_chain


# ------------------------------
# 전역변수/세션관리
# ------------------------------
ADMIN_CODE = "musai123"
save_dir = "rag_resources/uploads"
if "admin_verified" not in st.session_state:    # 관리자 승인 여부(테스트시 True)
    st.session_state.admin_verified = True
if "test_mode" not in st.session_state:         # 테스트 모드 플래그
    st.session_state.test_mode = False
if "uploaded_filename" not in st.session_state: # 현재 테스트 중 파일명
    st.session_state.uploaded_filename = ""
if "file_structured" not in st.session_state:   # 업로드파일 구조화 여부(표)
    st.session_state.file_structured = False
if "vectordb" not in st.session_state:          # 현재 vectordb
    st.session_state.vectordb = ""
if "uploader_version" not in st.session_state:  # file_uploader 초기화 key
    st.session_state.uploader_version = 0

# ------------------------------
# 관리자 - rag 생성
# ------------------------------
def creatVectordb(file_url):
    with st.spinner("임베딩 진행 중..."):
        success, vectordb = rag_chain.get_vectordb(file_url, st.session_state.file_structured)
        if success:
            st.success("벡터DB 생성 성공!")
            st.session_state.vectordb = vectordb
        else:
            st.error("벡터DB 생성 실패!")


def adminChunkRag(file_url):
    if st.session_state.admin_verified and st.session_state.test_mode:
        with st.spinner("청킹 진행 중..."):
            etl_data = rag_chain.get_chunked_docs(file_url, st.session_state.file_structured)
            
        st.success("ETL 완료! 아래는 청킹 결과입니다.")
        chunks = etl_data

        st.subheader("🧩 Chunked Documents")
        for idx, doc in enumerate(chunks):
            with st.expander(f"Chunk {idx+1}", expanded=False):
                st.markdown(f"**📄 Source:** `{doc.metadata.get('source', 'N/A')}`")
                st.markdown("---")
                st.text(doc.page_content)

def saveVectordb():
    if st.session_state.admin_verified and st.session_state.test_mode and st.session_state.vectordb :
        with st.spinner("데이터베이스 저장 중..."):
            success = rag_chain.set_vectordb(os.path.join(save_dir, st.session_state.uploaded_filename), st.session_state.file_structured)
            if success:
                st.success("저장 성공!")
            else:
                st.error("저장 실패!")

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

        if st.toggle("구조화"):
            st.session_state.file_structured = True

        # 테스트 버튼
        if st.button("테스트", key="btn_test", type="primary"):
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
        
        if st.button("🧩 청킹 실행"):
            adminChunkRag(file_url)

        if st.button("🧬 임베딩 실행"):
            creatVectordb(file_url)


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
    
    st.button(
        "save",
        icon=":material/save:",
        on_click=saveVectordb,
        use_container_width=True
    )
        

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