import config.setting
import json
from loguru import logger
from llm.models import get_llm
from prompts.load_prompt import get_prompt
from etl.langchain_parsers import clean_llm_json, parse_structured_json
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
    ".structured": RecursiveCharacterTextSplitter,
}

# 기본 Splitter (fallback)
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter

def chunk_data(documents, ext: str):
    logger.info(f"Start chunk_data : ext={ext}, documents={len(documents)}")
    """
    파일 확장자에 따라 적절한 Splitter를 선택하여 chunking을 수행.
    """
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
        logger.success(f"Done chunk_data : total_chunks={len(chunks)}")

        return chunks

    except Exception:
        logger.exception(f"Failed chunk_data: {ext}")
        raise


def chunk_structured_by_llm(chunked_docs):
    logger.info(f"Start chunk_structured_by_llm: chunked_docs={len(chunked_docs)}")

    docs_text = "\n".join(
        cd.page_content for cd in chunked_docs if cd.page_content
    )
    
    # docs_text = docs_text[:8000] # 비용 방어 

    # LLM에게 구조화 감지 요청
    llm = get_llm("gpt-4.1-mini")
    detect_structures_prompt = get_prompt("detect_structures.txt")
    detect_chain = detect_structures_prompt | llm
    detected_response = clean_llm_json(detect_chain.invoke({"docs_text": docs_text}))

    structures = []
    try:
        structures = json.loads(detected_response).get("structures", [])
        logger.success(f"find structures: detected_structures={len(structures)}")
    except Exception:
        logger.exception(f"find structures failed : {Exception}")
        raise

    # 4. confidence filter (구조화 확률 검증)
    structures = [
        s for s in structures
        if s.get("confidence", 0) >= 0.6
    ]

    EXTRACTOR_MAP = {
        "table": "extract_table.txt",
        "qa_pairs": "extract_qa_pairs.txt",
        "list": "extract_list.txt",
        "definition": "extract_definition.txt",
        "timeline": "extract_timeline.txt",
        "spec": "extract_spec.txt",
    }

    # 구조화 목록 생성
    structured_docs = []
    for s in structures:
        stype = s["type"]
        
        if stype not in EXTRACTOR_MAP:
            continue

        logger.debug(f"[stype] {stype} && [EXTRACTOR_MAP] {EXTRACTOR_MAP[stype]}")

        extract_prompt = get_prompt(EXTRACTOR_MAP[stype])
        extract_chain = extract_prompt | llm
        extract_response = clean_llm_json(extract_chain.invoke({"content": s["content"]}))
        #Document 형태로 변환
        structured_docs.extend(parse_structured_json(extract_response))
        

    logger.debug(f"[structured_docs] {structured_docs}")
    chunked_structured_docs = chunk_data(structured_docs, ".structured")
    
    logger.success(
        f"Done chunk_structured_by_llm : chunked_structured_docs={len(chunked_structured_docs)}"
    )

    return chunked_structured_docs
