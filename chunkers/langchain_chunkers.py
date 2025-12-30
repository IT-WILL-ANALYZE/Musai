import config.setting
import json
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
MAX_BLOCK_CHARS = 1200   # SemanticBlock 최대 허용 길이
SEMANTIC_OVERLAP = 100  # 의미 보존용

def chunk_data(documents, ext: str):
    logger.info(f"Start chunk_data : ext={ext}, documents={len(documents)}")

    splitter_class = SPLITTERS.get(ext, DEFAULT_SPLITTER)
    logger.info(f"Using splitter: {splitter_class.__name__}")

    collected_docs = []
    llm = get_llm("gpt-4.1-mini")

    for doc in documents:
        category = doc.metadata.get("category")
        base_index = float(doc.metadata.get("index", 0))

        # -----------------------------
        # 1️⃣ NO-CHUNK CATEGORY
        # -----------------------------
        if category in NO_CHUNK_CATEGORIES or category == "llm_friendly_table":
            collected_docs.append(doc)

            # Table → LLM 친화 Table 생성
            if category == "Table":
                try:
                    prompt = get_prompt("chunk_table.txt")
                    chain = prompt | llm
                    llm_response = chain.invoke(
                        {"text": doc.page_content}
                    ).content.strip()

                    collected_docs.append(
                        Document(
                            page_content=llm_response,
                            metadata={
                                **doc.metadata,
                                "index": base_index + 0.01,
                                "parent_index": base_index,
                                "content_type": "llm_friendly_table",
                            }
                        )
                    )
                except Exception:
                    logger.exception("chunk_table_by_llm failed")

            continue

        # -----------------------------
        # 2️⃣ SemanticBlock 전용 전략
        # -----------------------------
        if category == "SemanticBlock":
            text = doc.page_content

            # 짧으면 그대로 유지
            if len(text) <= MAX_BLOCK_CHARS:
                collected_docs.append(doc)
                continue

            # 🔹 길면 의미 단위 분할
            splitter = splitter_class(
                chunk_size=MAX_BLOCK_CHARS,
                chunk_overlap=SEMANTIC_OVERLAP,
            )
            chunks = splitter.split_text(text)

            for i, chunk_text in enumerate(chunks):
                collected_docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "index": base_index + (i * 0.001),
                            "parent_index": base_index,
                            "semantic_split": True,
                        }
                    )
                )
            continue

        # -----------------------------
        # 3️⃣ 기타 일반 텍스트 (기존 방식)
        # -----------------------------
        splitter = splitter_class(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents([doc])

        for i, ch in enumerate(chunks):
            ch.metadata["index"] = base_index + (i * 0.001)
            ch.metadata["parent_index"] = base_index
            collected_docs.append(ch)

    # -----------------------------
    # 안전장치
    # -----------------------------
    safe_docs = []
    for d in collected_docs:
        if isinstance(d, Document):
            safe_docs.append(d)
        else:
            logger.error(
                f"Non-Document dropped in chunk_data: type={type(d)} value={str(d)[:80]}"
            )

    safe_docs.sort(key=lambda d: float(d.metadata.get("index", 0)))

    logger.info(f"Done chunk_data : total_docs={len(safe_docs)}")
    return safe_docs



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
