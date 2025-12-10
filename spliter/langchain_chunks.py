from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)


# 확장자별 Splitter 매핑
SPLITTERS = {
    ".md": MarkdownTextSplitter,     # Markdown 전용 Splitter
    ".markdown": MarkdownTextSplitter,
    ".txt": CharacterTextSplitter,   # 매우 단순한 텍스트는 Character 기반
    ".pdf": RecursiveCharacterTextSplitter,  # PDF는 Recursive가 안정적
    ".docx": RecursiveCharacterTextSplitter,
    ".csv": CharacterTextSplitter,   # CSV는 보통 Character 기반이 안전
    ".xlsx": RecursiveCharacterTextSplitter,
    ".xls": RecursiveCharacterTextSplitter,
    # ".hwp": ??? → HWP 추가 예정
}

# 기본 Splitter (확장자 매핑에 없을 경우)
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter

def chunk_data(documents, ext: str):
    """
    파일 확장자에 따라 다른 Splitter를 자동 선택하여
    문서를 chunking 한다.
    extension 예: '.md', '.pdf', '.txt'
    """

    # Splitter 선택 (fallback: DEFAULT_SPLITTER)
    splitter_class = SPLITTERS.get(ext, DEFAULT_SPLITTER)

    # splitter 인스턴스 생성
    splitter = splitter_class( #청크 범위와 길이는 고정으로 하는것이 안정적 
        chunk_size=500,
        chunk_overlap=50
    )

    # 문서 chunking
    return splitter.split_documents(documents)

