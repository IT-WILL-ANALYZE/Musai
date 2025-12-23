from loguru import logger
import config.setting as config
from langchain_core.messages import AIMessage
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
    logger.info(f"Start load_by_langchain : {file_url, mode, strategy}")

    cfg = VALID_LOADERS_SETTINGS[ext]
    loader_class = cfg["LOADER"]

    mode, strategy = normalize_loader_options(ext, mode, strategy)
    logger.info(f"set loader : [ext] {ext} || [mode] {mode} || [strategy] {strategy}" )
    
    loader_kwargs = {}
    if mode is not None:
        loader_kwargs["mode"] = mode
    if strategy is not None:
        loader_kwargs["strategy"] = strategy
        if strategy == "hi_res" :
            poppler_path = config.get_bin_path("poppler")   # 이미지로 변경
            config.get_bin_path("tesseract")                # 이미지에서 단어 및 형태 추출
            loader_kwargs["poppler_path"] = poppler_path 
            loader_kwargs["ocr_languages"] = config.get_bin_path("kor+eng")

    loader = loader_class(file_url, **loader_kwargs)
    docs = loader.load()

    # 2. 데이터 분류를 위한 리스트 준비
    text_docs = []       # 일반 텍스트 (category 없거나 Title, NarrativeText 등)
    structured_docs = [] # 구조화 데이터 (Table, Figure 등)

    # 3. Category 기반 분류 로직
    for doc in docs:
        category = doc.metadata.get("category")
        
        # 1. 구조화/정제 대상 데이터
        if category in ["Table", "Figure", "Image", "UncategorizedText"]:
            structured_docs.append(doc)
        
        # 2. 일반 텍스트 데이터
        else:
            text_docs.append(doc)

    logger.info(f"Done: load_by_langchain Text({len(text_docs)}) || Structured({len(structured_docs)})")
    
    return text_docs, structured_docs


