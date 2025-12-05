from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 허용 가능한 모델 목록
ALLOWED_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-pro",
    # 필요한 모델을 여기에 추가
]

def set_model(model: str) -> ChatOpenAI:
    """
    주어진 모델명이 허용된 모델 목록에 없으면 기본 모델(gpt-4.1-mini)로 설정하여
    ChatOpenAI 인스턴스를 반환합니다.
    """
    llm_model = model if model in ALLOWED_MODELS else "gpt-4.1-mini"
    return ChatOpenAI(model=llm_model)