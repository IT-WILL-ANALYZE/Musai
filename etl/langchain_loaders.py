from loguru import logger
import config.setting as config
from bs4 import BeautifulSoup
from etl.langchain_parsers import clean_llm_json
from langchain_core.documents import Document
from typing import List
import json
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)


class UTF8TextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding="utf-8")


# --- Loader 매핑 ---
VALID_LOADERS_SETTINGS = {
    ".md": {
        "LOADER": UnstructuredMarkdownLoader,
        "MODES": {"elements", "single", "paged"},
        "STRATEGIES": set(),   # strategy 의미 없음
    },
    ".markdown": {
        "LOADER": UnstructuredMarkdownLoader,
        "MODES": {"elements", "single", "paged"},
        "STRATEGIES": set(),
    },
    ".txt": {
        "LOADER": UTF8TextLoader,
        "MODES": set(),
        "STRATEGIES": set(),
    },
    ".pdf": {
        "LOADER": UnstructuredPDFLoader,
        "MODES": {"elements", "single", "paged"},
        "STRATEGIES": {"fast", "hi_res"}, #ocr_only, auto
    },
    ".docx": {
        "LOADER": UnstructuredWordDocumentLoader,
        "MODES": {"elements", "single"},
        "STRATEGIES": set(),
    },
    ".csv": {
        "LOADER": CSVLoader,
        "MODES": set(),        # loader가 mode 자체를 안 씀
        "STRATEGIES": set(),
    },
    ".xlsx": {
        "LOADER": UnstructuredExcelLoader,
        "MODES": {"elements"},
        "STRATEGIES": set(),
    },
    ".xls": {
        "LOADER": UnstructuredExcelLoader,
        "MODES": {"elements"},
        "STRATEGIES": set(),
    },
}


# ----------------------------------------------------
# 확장자 검사
# ----------------------------------------------------
def get_ext_from_filename(file_url: str):
    ext = "." + file_url.lower().split(".")[-1]
    return ext


# ----------------------------------------------------
# 옵션 정규화
# ----------------------------------------------------
def normalize_loader_options(ext: str, mode=None, strategy=None):
    cfg = VALID_LOADERS_SETTINGS[ext]

    # --- mode ---
    valid_modes = cfg["MODES"]
    if valid_modes:
        mode = mode if mode in valid_modes else next(iter(valid_modes))
    else:
        mode = None

    # --- strategy ---
    valid_strategies = cfg["STRATEGIES"]
    if valid_strategies:
        strategy = strategy if strategy in valid_strategies else "fast"
    else:
        strategy = None

    return mode, strategy


# ----------------------------------------------------
# HTML → Markdown 변환
# ----------------------------------------------------
def html_table_to_markdown(html: str) -> str:
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return ""

        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [cell.get_text(strip=True) for cell in cells]
            if row:
                rows.append(row)

        if len(rows) < 2:
            return ""

        header = rows[0]
        body = rows[1:]

        md = []
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in body:
            md.append("| " + " | ".join(row) + " |")

        return "\n".join(md)

    except Exception as e:
        logger.warning(f"[html_table_to_markdown] failed: {e}")
        return ""
    
    
def merge_docs_by_categories(docs: List[Document]) -> List[Document]:
    CONTINUE_CATEGORIES = {"TableChunk", "UncategorizedText"}
    merged_docs: List[Document] = []

    buffer: List[Document] = []
    buffer_category = None
    index = 0

    def flush_buffer():
        nonlocal index, buffer, buffer_category, merged_docs

        if not buffer:
            return

        # -----------------------
        # TableChunk → Table 병합
        # -----------------------
        if buffer_category in ("TableChunk", "Table"):
            raw_text = "\n".join(doc.page_content for doc in buffer)

            html = buffer[0].metadata.get("text_as_html", "")
            markdown_table = html_table_to_markdown(html)

            final_content = (
                markdown_table if markdown_table else raw_text
            )

            # 불필요한 대용량 필드 제거
            clean_metadata = {
                k: v for k, v in buffer[0].metadata.items()
                if k not in {"orig_elements", "text_as_html"}
            }

            merged_docs.append(
                Document(
                    page_content=final_content,
                    metadata={
                        **clean_metadata,
                        "category": "Table",
                        "merged_from": "TableChunk",
                        "index": index,
                        "table_format": "markdown" if markdown_table else "text",
                    },
                )
            )
            index += 1

        # -----------------------
        # UncategorizedText → LLM 분류
        # -----------------------
        elif buffer_category == "UncategorizedText":
            
            llm = get_llm("gpt-4.1-mini")
            prompt = get_prompt("UNCATEGORIZED_CLASSIFY.txt")

            block_text = "\n".join(doc.page_content for doc in buffer)[:4000]

            try:
                response = clean_llm_json(
                    (prompt | llm).invoke({"text": block_text})
                )
                Categorized_json = json.loads(response)
                logger.info(f"Uncategorized by LLM : {buffer_category} → {Categorized_json.get("cleaned_category", "UncategorizedText")} || buffer size : {len(buffer)}")

                merged_docs.append(
                    Document(
                        page_content=Categorized_json.get("cleaned_text", block_text),
                        metadata={
                            **buffer[0].metadata,
                            "category": Categorized_json.get("cleaned_category", "UncategorizedText"),
                            "index": index,
                            "classified_by": "llm",
                        },
                    )
                )
                index += 1

            except Exception as e:
                logger.warning(f"Uncategorized by LLM failed: {e}")

                merged_docs.append(
                    Document(
                        page_content=block_text,
                        metadata={
                            **buffer[0].metadata,
                            "category": "UncategorizedText",
                            "index": index,
                        },
                    )
                )
                index += 1

        buffer = []
        buffer_category = None

    # -----------------------
    # Main loop
    # -----------------------
    for doc in docs:
        category = doc.metadata.get("category")

        if category in CONTINUE_CATEGORIES:
            if buffer_category in (None, category):
                buffer.append(doc)
                buffer_category = category
            else:
                flush_buffer()
                buffer.append(doc)
                buffer_category = category
        else:
            flush_buffer()
            doc.metadata["index"] = index
            merged_docs.append(doc)
            index += 1

    flush_buffer()
    return merged_docs

# ----------------------------------------------------
# 문서 로드
# ----------------------------------------------------
def load_by_langchain(file_url: str, mode=None, strategy=None):
    ext = get_ext_from_filename(file_url)
    logger.info(f"Start load_by_langchain : {file_url}")

    cfg = VALID_LOADERS_SETTINGS[ext]
    loader_class = cfg["LOADER"]

    mode, strategy = normalize_loader_options(ext, mode, strategy)
    logger.info(f"Loader options → ext={ext}, mode={mode}, strategy={strategy}")

    loader_kwargs = {}
    if mode:
        loader_kwargs["mode"] = mode
    if strategy:
        loader_kwargs["strategy"] = strategy
        if strategy == "hi_res":
            poppler_path = config.get_bin_path("poppler")
            config.get_bin_path("tesseract")                # 이미지에서 단어 및 형태 추출
            loader_kwargs["poppler_path"] = poppler_path 
            loader_kwargs["ocr_languages"] = ["kor", "eng"]
            loader_kwargs["infer_table_structure"] =True,   # 테이블 구조 분석 활성화
            loader_kwargs["chunking_strategy"] = "by_title" # 테이블 타이블 활성화
    
    loader = loader_class(file_url, **loader_kwargs)
    docs = loader.load()

    # -----------------------------
    # 구조화 및 병합
    # -----------------------------
    merged_docs = merge_docs_by_categories(docs)

    logger.debug(f"return load_by_langchain content[:3]={merged_docs[:3]}")
    logger.info(f"Done load_by_langchain : len = {len(merged_docs)}")
    return merged_docs




