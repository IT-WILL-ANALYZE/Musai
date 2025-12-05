from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)


# 확장자 → Loader 매핑
LOADERS = {
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".pdf": UnstructuredPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    #".hwp": HwpLoader,  # 한글 파일 -> 추가예정
}

# 허용 가능한 옵션
VALID_MODES = {"elements", "single", None}
VALID_STRATEGIES = {"fast", "hi_res", "ocr_only", "auto", None}

def check_data(file_url: str):
    
    ext = "." + file_url.lower().split(".")[-1]

    if ext not in LOADERS:
        return {
            "status": "error",
            "message": f"Unsupported file extension: {ext}"
        }

    return {
        "status": "ok",
        "extension": ext
    }


def load_data(file_url: str, mode=None, strategy=None):
    """
    mode와 strategy를 직접 입력받되,
    사용자가 잘못된 값을 넣으면 기본값으로 설정한다.
    기본 mode="elements", strategy="fast".
    """
    # 확장자 검사
    check_result = check_data(file_url)
    if check_result["status"] == "error":
        return check_result
    
    # 기본값 처리
    mode = mode if mode in VALID_MODES else "elements"
    strategy = strategy if strategy in VALID_STRATEGIES else "fast"

    # 확장자 추출
    ext = "." + file_url.lower().split(".")[-1]

    # 로더 선택 (fallback: UnstructuredFileLoader)
    loader_class = LOADERS.get(ext, UnstructuredFileLoader)

    # Unstructured 계열만 mode/strategy 적용
    if "Unstructured" in loader_class.__name__:
        loader = loader_class(
            file_url,
            mode=mode,
            strategy=strategy
        )
    else:
        # TextLoader, CSVLoader 등은 mode/strategy 옵션 없음
        loader = loader_class(file_url)

    return loader.load()