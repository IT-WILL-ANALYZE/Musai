import config.setting
from langchain_openai import ChatOpenAI



# 허용 가능한 모델 목록
ALLOWED_MODELS = [
    # GPT-5.2 계열 (최신)
    "gpt-5.2",
    "gpt-5.2-chat-latest",
    "gpt-5.2-pro",
    "gpt-5.2-instant",
    "gpt-5.2-thinking",

    # GPT-5 계열
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-pro",

    # GPT-4.1 계열
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-pro",

    # GPT-4o 계열
    "gpt-4o",
    "gpt-4o-mini",

    # GPT-4 Turbo / Legacy
    "gpt-4-turbo",
    "gpt-4",

    # GPT-3.5 (비용 절감용)
    "gpt-3.5-turbo",
]

def get_llm(model: str) -> ChatOpenAI:
    """
    주어진 모델명이 허용된 모델 목록에 없으면 기본 모델(gpt-4.1-mini)로 설정하여
    ChatOpenAI 인스턴스를 반환합니다.
    """
    llm_model = model if model in ALLOWED_MODELS else "gpt-4.1-mini"
    return ChatOpenAI(model=llm_model)