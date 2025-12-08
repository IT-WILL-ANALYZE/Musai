import streamlit as st
import os as os
# import chain as chain


# ------------------------------
# 전역변수/세션관리
# ------------------------------
ADMIN_CODE = "musai123"
save_dir = "original_data"
if "admin_verified" not in st.session_state:    # 관리자 승인 여부
    st.session_state.admin_verified = False
if "test_mode" not in st.session_state:         # 테스트 모드 플래그
    st.session_state.test_mode = False
if "uploaded_filename" not in st.session_state: # 현재 테스트 중 파일명
    st.session_state.uploaded_filename = ""
if "uploader_version" not in st.session_state:  # file_uploader 초기화 key
    st.session_state.uploader_version = 0
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
        st.session_state.uploaded_filename = uploaded.name
        save_path = os.path.join(save_dir, uploaded.name)

        st.success(f"파일 선택됨: {uploaded.name}")

        if os.path.exists(save_path):
            st.warning(f"⚠ '{uploaded.name}' 파일은 이미 존재합니다. 테스트 시 덮어씌워집니다.")

        # 테스트 버튼
        if st.button("테스트", type="primary"):

            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.session_state.test_mode = True
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
    left, spacer, right = st.columns([1, 6, 1])

    with left:
        st.button(
            "&nbsp;:small[:gray[:material/Cognition: MusAi란]]",
            type="tertiary",
            on_click=show_who_am_i,
            use_container_width=True,
        )

    with right:
        st.button(
            "&nbsp;:small[:gray[:material/Face: 관리자모드]]"
            if not st.session_state.admin_verified
            else "&nbsp;:small[:gray[:material/upload: 파일업로드]]",
            type="tertiary",
            on_click=show_upload_file,
            use_container_width=True,
        )

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

# chat_input 에서 바로 입력이 안 들어온 경우
# (ex. 첫 진입 시 initial_question / suggestion 으로 들어온 경우)
if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    elif user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

# 실제로 처리할 메시지가 존재하는 경우
if user_message:
    # 한 번 사용한 initial/suggestion 상태 초기화 (중복 처리 방지)
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None

    # 유저 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_message})

    # --- 여기서 LLM 호출이 들어갈 자리 ---
    dummy_reply = (
        "(여기에는 AI 응답이 표시될 자리입니다)\n\n"
        f"You said: {user_message}"
    )

    # 어시스턴트 응답 저장
    st.session_state.messages.append(
        {"role": "assistant", "content": dummy_reply}
    )

    # 새 메시지 반영 위해 리렌더
    st.rerun()
