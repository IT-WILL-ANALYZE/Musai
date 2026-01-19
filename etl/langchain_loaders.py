from loguru import logger
import config.setting as config
import base64
import os
import pymupdf4llm
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
        "CONVERTER" : set(),
    },
    ".markdown": {
        "LOADER": UnstructuredMarkdownLoader,
        "MODES": {"elements", "single", "paged"},
        "STRATEGIES": set(),
        "CONVERTER" : set(),
    },
    ".txt": {
        "LOADER": UTF8TextLoader,
        "MODES": set(),
        "STRATEGIES": set(),
        "CONVERTER" : "text_to_md",
    },
    ".pdf": {
        "LOADER": UnstructuredPDFLoader,
        "MODES": {"elements", "single", "paged"},
        "STRATEGIES": {"fast"}, # except hi_res
        "CONVERTER" : "pdf_to_md",
    },
    ".docx": {
        "LOADER": UnstructuredWordDocumentLoader,
        "MODES": {"elements", "single"},
        "STRATEGIES": set(),
        "CONVERTER" : "text_to_md",
    },
    ".csv": {
        "LOADER": CSVLoader,
        "MODES": set(),        # loader가 mode 자체를 안 씀
        "STRATEGIES": set(),
        "CONVERTER" : "table_to_md",
    },
    ".xlsx": {
        "LOADER": UnstructuredExcelLoader,
        "MODES": {"elements"},
        "STRATEGIES": set(),
        "CONVERTER" : "table_to_md",
    },
    ".xls": {
        "LOADER": UnstructuredExcelLoader,
        "MODES": {"elements"},
        "STRATEGIES": set(),
        "CONVERTER" : "table_to_md",
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
# 확장자별 마크다운 변환 핵심 로직
# ----------------------------------------------------
def _convert_to_markdown(docs, file_url, converter_type):
    """
    이미지 분석 후 메모리/디스크 최적화를 수행하며, 
    추출된 마크다운을 page_content에 직접 반영합니다.
    """
    logger.info(f"Converting docs using strategy: {converter_type}")
    
    structured_docs = []
    
    # 1. PyMuPDF4LLM을 사용하여 전체 텍스트를 마크다운으로 변환
    # (full_md_context를 메타데이터에 넣지 않고 직접 활용)
    full_markdown_text = ""
    if converter_type == "pdf_to_md":
        try:
            full_markdown_text = pymupdf4llm.to_markdown(file_url)
        except Exception as e:
            logger.error(f"PyMuPDF4LLM 변환 실패: {e}")

    # 2. LLM 및 프롬프트 준비
    llm = get_llm("gpt-4.1-mini")
    detect_structures_prompt = get_prompt("detect_structures.txt")
    detect_chain = detect_structures_prompt | llm

    # 3. 요소별 처리
    for doc in docs:
        original_category = doc.metadata.get("category", "Text")
        page_content = doc.page_content
        
        new_metadata = {
            **doc.metadata,
            "category": original_category,
            "categorized_with_llm": None,
        }

        # --- 이미지/그림 요소인 경우에만 LLM 분석 수행 ---
        is_image_element = original_category in ["Image", "Figure", "Graphic"]
        
        if converter_type == "pdf_to_md" and is_image_element:
            image_path = doc.metadata.get("image_path")
            
            if image_path and os.path.exists(image_path):
                base64_raw = None # 초기화
                try:
                    # [파일 -> Base64 생성]
                    with open(image_path, "rb") as f:
                        base64_raw = base64.b64encode(f.read()).decode("utf-8")
                        formatted_base64 = f"data:image/png;base64,{base64_raw}"

                    # [LLM 분석 작업 진행]
                    logger.info(f"LLM 이미지 분석 중... (Page: {doc.metadata.get('page_number')})")
                    raw_response = detect_chain.invoke({"image": formatted_base64})
                    detected_data = clean_llm_json(raw_response)
                    
                    if isinstance(detected_data, dict):
                        # 분석된 내용을 page_content에 삽입
                        page_content = detected_data.get("content", page_content)
                        new_metadata["categorized_with_llm"] = detected_data.get("category")
                
                except Exception as e:
                    logger.error(f"이미지 분석 중 오류 발생: {e}")
                    new_metadata["error"] = str(e)
                
                finally:
                    # [메모리 최적화] base64 변수 초기화
                    base64_raw = None
                    # [정상 종료/에러 공통] 임시 이미지 파일 즉시 삭제
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        logger.debug(f"임시 이미지 삭제 완료: {image_path}")

        # 가공된 Document 객체 생성
        structured_docs.append(Document(
            page_content=page_content,
            metadata=new_metadata
        ))

    if full_markdown_text:
        md_doc = Document(
            page_content=full_markdown_text,
            metadata={"category": "Full_Markdown", "source": file_url}
        )
        # 리스트 맨 앞에 삽입하여 검색 시 전체 맥락이 먼저 고려되도록 함
        structured_docs.insert(0, md_doc)
    
    return structured_docs

# ----------------------------------------------------
# 문서 로드
# ----------------------------------------------------
def load_by_langchain(file_url: str, mode=None, strategy=None):
    ext = get_ext_from_filename(file_url)
    logger.info(f"Start load_by_langchain : {file_url}")

    cfg = VALID_LOADERS_SETTINGS[ext]
    loader_class = cfg["LOADER"]
    converter_type = cfg["CONVERTER"]

    mode, strategy = normalize_loader_options(ext, mode, strategy)
    logger.info(f"Loader options → ext={ext}, mode={mode}, strategy={strategy}")

    loader_kwargs = {}
    if mode:
        loader_kwargs["mode"] = mode
    if strategy:
        loader_kwargs["strategy"] = strategy
    
    loader = loader_class(file_url, **loader_kwargs)
    docs = loader.load()
    
    # -----------------------------
    # 구조화 및 병합
    # -----------------------------
    converted_docs = _convert_to_markdown(docs, file_url, converter_type)

    logger.info(f"Done load_by_langchain : len = {len(converted_docs)}")
    return converted_docs


'''
참고 )langchain을 통한 OCR : poppler - ocr //1 tesseract - image read
loader_kwargs = {}
loader_kwargs["mode"] = mode
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
'''



