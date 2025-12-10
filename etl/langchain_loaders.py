import json
import os
from datetime import datetime
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
        
# 확장자 → Loader 매핑
LOADERS = {
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".txt": UTF8TextLoader,
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


def load_data(file_url: str, ext: str, mode=None, strategy=None):
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

    # 로더 선택 (fallback: UnstructuredFileLoader)
    loader_class = LOADERS.get(ext, UnstructuredFileLoader)

    if ext == ".docx":
        loader = loader_class(file_url)  # 옵션 없이 반드시 이렇게!
        return loader.load()
    # Unstructured 계열만 mode/strategy 적용
    elif "Unstructured" in loader_class.__name__:
        loader = loader_class(
            file_url,
            mode=mode,
            strategy=strategy
        )
    else:
        # TextLoader, CSVLoader 등은 mode/strategy 옵션 없음
        loader = loader_class(file_url)

    return loader.load()


    
def save_knowledgebase(file_url, chunked_docs, vector_ids):
    
    # file_url → file_name 추출 (확장자 제거)
    base_name = os.path.splitext(os.path.basename(file_url))[0]

    base_path = "rag_resources/knowledge-base"
    os.makedirs(base_path, exist_ok=True)

    # 중복 검사 후 최종 file_name 결정
    file_name = base_name
    counter = 1

    while os.path.exists(os.path.join(base_path, f"{file_name}.json")):
        file_name = f"{base_name}_{counter}"
        counter += 1

    # JSON 구조 생성
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

    # 파일 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return save_path