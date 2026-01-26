import config.setting
import json
import re
from loguru import logger
from langchain_core.documents import Document
from llm.models import get_llm
from prompts.load_prompt import get_prompt
from etl.langchain_parsers import clean_llm_json, parse_structured_json
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)



# ----------------------------------------------------
# 메인 청크(현재 사용 X)
# ----------------------------------------------------

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

# 청킹하지 않는 구조
NO_CHUNK_CATEGORIES = {"Table", "Figure", "Image", "UncategorizedText"}

def chunk_data(docs: list[Document], ext: str):
    logger.info(f"Start chunk_data : ext={ext}, documents={len(docs)}")

    splitter_class = SPLITTERS.get(ext, DEFAULT_SPLITTER)
    logger.info(f"Using splitter: {splitter_class.__name__}")

    splitter = splitter_class(
        chunk_size=800,
        chunk_overlap=100,
    )

    results = []

    for doc in docs:
        category = doc.metadata.get("category")

        # Table / Figure 등은 청킹 스킵
        if category in NO_CHUNK_CATEGORIES:
            logger.info(f"Skip chunking for category={category}")
            results.append(doc)
            continue

        # 일반 텍스트만 청킹
        chunks = splitter.split_documents([doc])
        results.extend(chunks)

    logger.debug(f"return chunk_data content[:3]={results[:3]}")
    logger.info(f"Done chunk_data : total_docs={len(results)}")
    return results


# ----------------------------------------------------
# 특정 확장자만 chunk(MD 확장자만 구현)
# ----------------------------------------------------

def chunk_format_md(content_md: str):
    logger.info(f"Start chunk_format_md : content_length={len(content_md)}")
    
    if not content_md:
        return []

    # MarkdownTextSplitter 설정
    text_splitter = MarkdownTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )

    # Document 객체로 변환 (메타데이터 포함)
    base_metadata = {
        "source": "loaded_content",
        "format": "markdown"
    }
    
    # create_documents를 사용하여 Document 객체 리스트 반환
    results = text_splitter.create_documents(
        [content_md],
        metadatas=[base_metadata]
    )
    
    # 각 Document에 category 메타데이터 추가
    for idx, doc in enumerate(results):
        doc.metadata["category"] = "Text"
        doc.metadata["chunk_index"] = idx + 1

    logger.info(f"Done chunk_format_md : total_chunks={len(results)}")
    return results


# ----------------------------------------------------
# 미구현
# ----------------------------------------------------

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
