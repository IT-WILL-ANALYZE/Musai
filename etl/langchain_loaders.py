import json
import os
from datetime import datetime
from loguru import logger
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

    ext = "." + file_url.lower().split(".")[-1]
    logger.debug(f"Checking extension for: {file_url}")

    if ext not in LOADERS:
        logger.error(f"Unsupported file extension: {ext}")
        return {
            "status": "error",
            "message": f"Unsupported file extension: {ext}"
        }

    return {"status": "ok", "extension": ext}


# ----------------------------------------------------
# 문서 로드
# ----------------------------------------------------
def load_data(file_url: str, ext: str, mode=None, strategy=None):
    """
    mode와 strategy 기본값을 직접 처리.
    기본 mode="elements", strategy="fast".
    """

    logger.info(f"Loading data: file={file_url}, ext={ext}, mode={mode}, strategy={strategy}")

    # 확장자 검사
    check_result = check_data(file_url)
    if check_result["status"] == "error":
        logger.error(f"Check data failed: {check_result['message']}")
        return check_result

    # 사용자가 잘못된 값을 넣으면 기본값 적용
    mode = mode if mode in VALID_MODES else "elements"
    strategy = strategy if strategy in VALID_STRATEGIES else "fast"

    loader_class = LOADERS.get(ext, UnstructuredFileLoader)

    try:
        # DOCX는 mode 적용 불가
        if ext == ".docx":
            loader = loader_class(file_url)
        # Unstructured 계열은 mode/strategy 사용
        elif "Unstructured" in loader_class.__name__:
            loader = loader_class(file_url, mode=mode, strategy=strategy)
        else:
            loader = loader_class(file_url)

        docs = loader.load()
        logger.success(f"Loaded {len(docs)} documents from {file_url}")
        return docs

    except Exception as e:
        logger.exception(f"Error loading data from file: {file_url}")
        raise


# ----------------------------------------------------
# Knowledgebase 저장
# ----------------------------------------------------
def save_knowledgebase(file_url, chunked_docs, vector_ids):

    logger.info(f"Saving knowledgebase for file: {file_url}")

    base_name = os.path.splitext(os.path.basename(file_url))[0]

    base_path = "rag_resources/knowledge-base"
    os.makedirs(base_path, exist_ok=True)

    # 중복된 이름 처리
    file_name = base_name
    counter = 1
    while os.path.exists(os.path.join(base_path, f"{file_name}.json")):
        file_name = f"{base_name}_{counter}"
        counter += 1

    data = {
        "meta": {
            "file_name": file_name,
            "total_chunks": len(chunked_docs),
            "created_at": datetime.now().isoformat()
        },
        "chunks": []
    }

    for i, doc in enumerate(chunked_docs):
        data["chunks"].append({
            "id": f"chunk_{i+1:03}",
            "content": doc.page_content,
            "metadata": doc.metadata,
            "vector_id": vector_ids[i] if vector_ids else None
        })

    save_path = os.path.join(base_path, f"{file_name}.json")

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.success(f"Knowledgebase saved: {save_path}")
        return save_path

    except Exception as e:
        logger.exception("Failed to save knowledgebase")
        raise
