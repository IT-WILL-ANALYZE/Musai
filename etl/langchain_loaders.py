from loguru import logger
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
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
LOADERS = {
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".txt": UTF8TextLoader,
    ".pdf": UnstructuredPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
}

# --- 옵션 범위 수정 ---
VALID_MODES = {"elements", "single", "paged"}
VALID_STRATEGIES = {"fast", "hi_res", "ocr_only", "auto"}


# ----------------------------------------------------
# 확장자 검사
# ----------------------------------------------------
def check_data(file_url: str):
    logger.info(f"Start check_data : {file_url}")

    ext = "." + file_url.lower().split(".")[-1]
    if ext not in LOADERS:
        logger.error(f"Failed check_data: {ext}")
        return {
            "status": "error",
            "message": f"Unsupported file extension: {ext}"
        }

    return {"status": "ok", "extension": ext}


# ----------------------------------------------------
# 문서 로드
# ----------------------------------------------------
def load_by_langchain(file_url: str, ext: str, mode=None, strategy=None):
    logger.info(f"Start load_by_langchain : {file_url, ext, mode, strategy}")
    """
    mode와 strategy 기본값을 직접 처리.
    기본 mode="elements", strategy="fast".
    """

    # 확장자 검사
    check_result = check_data(file_url)
    if check_result["status"] == "error":
        return check_result

    # 사용자가 잘못된 값을 넣으면 기본값 적용
    mode = mode if mode in VALID_MODES else "elements"
    strategy = strategy if strategy in VALID_STRATEGIES else "fast"
    loader_class = LOADERS.get(ext, UnstructuredFileLoader)

    
    # DOCX는 mode 적용 불가
    if ext == ".docx":
        loader = loader_class(file_url)
    # Unstructured 계열은 mode/strategy 사용
    elif "Unstructured" in loader_class.__name__:
        loader = loader_class(file_url, mode=mode, strategy=strategy)
    else:
        loader = loader_class(file_url, mode=mode)

    docs = loader.load()
    
    logger.success(f"Done load_by_langchain : {len(docs)}")
    return docs
