from loguru import logger
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)

# 확장자별 Splitter 매핑
SPLITTERS = {
    ".md": MarkdownTextSplitter,
    ".markdown": MarkdownTextSplitter,
    ".txt": CharacterTextSplitter,
    ".pdf": RecursiveCharacterTextSplitter,
    ".docx": RecursiveCharacterTextSplitter,
    ".csv": CharacterTextSplitter,
    ".xlsx": RecursiveCharacterTextSplitter,
    ".xls": RecursiveCharacterTextSplitter,
}

# 기본 Splitter (fallback)
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter


def chunk_data(documents, ext: str):
    """
    파일 확장자에 따라 적절한 Splitter를 선택하여 chunking을 수행.
    """

    logger.info(f"Starting chunking: ext={ext}, documents={len(documents)}")

    # Splitter 선택
    splitter_class = SPLITTERS.get(ext, DEFAULT_SPLITTER)
    logger.debug(f"Using splitter: {splitter_class.__name__}")

    try:
        # Splitter 인스턴스 생성
        splitter = splitter_class(
            chunk_size=500,
            chunk_overlap=50
        )

        # chunk 실행
        chunks = splitter.split_documents(documents)
        logger.success(f"Chunking completed: total_chunks={len(chunks)}")

        return chunks

    except Exception:
        logger.exception(f"Chunking failed for extension: {ext}")
        raise
