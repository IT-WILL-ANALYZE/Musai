from loguru import logger
import config.setting as config
from etl.langchain_parsers import clean_llm_json
from langchain_core.documents import Document
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
            loader_kwargs["ocr_languages"] = config.get_bin_path("kor+eng")

    loader = loader_class(file_url, **loader_kwargs)
    raw_docs = loader.load()

    # -----------------------------
    # 설정
    # -----------------------------
    CONTEXT_CATEGORIES = {"Title", "NarrativeText", "List", "Caption"}
    STRUCTURED_CATEGORIES = {"Table", "Figure", "Image"}

    llm = get_llm("gpt-4.1-mini")

    merged_docs = []
    index = 0
    current_title = None
    current_block = []
    current_categories = []

    # -----------------------------
    # 내부 함수: 블록 flush
    # -----------------------------
    def flush_block():
        nonlocal index, current_block, current_categories, current_title

        if not current_block:
            return

        block_text = "\n".join(current_block)
        final_category = "SemanticBlock"
        inferred_title = current_title
        if not inferred_title:
            # title이 아예 없으면 블록 첫 문장을 title처럼 사용(너무 길면 자름)
            inferred_title = (current_block[0][:80] if current_block else None)

        if "UncategorizedText" in current_categories:
            prompt = get_prompt("UNCATEGORIZED_CLASSIFY.txt")
            response = clean_llm_json(
                (prompt | llm).invoke({
                    "title": inferred_title,
                    "text": block_text[:4000]
                })
            )
            try:
                result = json.loads(response)
                logger.info(f"[flush_block] LLM response {result}")
                final_category = result.get("category", final_category)
                block_text = result.get("cleaned_text", block_text)
            except Exception as e:
                logger.warning(f"[flush_block] LLM classify failed: {e}")

        merged_docs.append(
            Document(
                page_content=block_text,
                metadata={
                    "source": file_url,
                    "title": inferred_title,
                    "category": final_category,
                    "included_categories": list(set(current_categories)),
                    "index": index,
                }
            )
        )

        index += 1
        current_block.clear()
        current_categories.clear()

    # -----------------------------
    # 메인 루프
    # -----------------------------
    for doc in raw_docs:
        category = doc.metadata.get("category", "UncategorizedText")
        text = doc.page_content.strip()

        # 1️⃣ Title 등장 → 새 의미 블록 시작
        if category == "Title":
            flush_block()
            current_title = text
            current_block.append(text)
            current_categories.append("Title")
            continue

        # 2️⃣ 구조화 데이터는 단독 문서
        if category in STRUCTURED_CATEGORIES:
            flush_block()
            merged_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_url,
                        "title": current_title,
                        "category": category,
                        "index": index,
                    }
                )
            )
            index += 1
            continue

        # 3️⃣ 일반 / Uncategorized → 현재 블록에 포함
        # Title이 아직 없는데 문맥성 카테고리면 fallback title로 사용
        if current_title is None and category in CONTEXT_CATEGORIES:
            current_title = text  # fallback title
        current_block.append(text)
        current_categories.append(category)

    # 마지막 블록 처리
    flush_block()

    logger.success(f"Done load_by_langchain : total_docs={len(merged_docs)}")
    return merged_docs




