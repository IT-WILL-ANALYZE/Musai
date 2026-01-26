import os
import base64
import fitz  # PyMuPDF
import pymupdf4llm
from loguru import logger
from typing import List
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from etl.langchain_parsers import clean_llm_json

class UTF8TextLoader(TextLoader):
    def __init__(self, file_path):
        super().__init__(file_path, encoding="utf-8")


def get_ext_from_filename(file_url: str):
    ext = os.path.splitext(file_url)[1].lower()
    return ext


# ----------------------------------------------------
# 메인 로드 함수
# ----------------------------------------------------

def load_by_langchain(file_url: str) -> str:
    """
    파일을 로드하여 마크다운 텍스트로 반환합니다.
    모든 형식(PDF, 이미지, 텍스트 등)을 마크다운으로 통일합니다.
    """
    logger.info(f"Start load_by_langchain : {file_url}")
    
    # 확장자 추출
    ext = os.path.splitext(file_url)[1].lower().replace('.', '')
    
    # 이미지 통합 처리 (.jpg, .png, .jpeg)
    if ext in ['jpg', 'jpeg', 'png']:
        text_content = extract_image(file_url)
    else:
        # 동적 함수 매핑 (extract_{ext})
        method_name = f"extract_{ext}"
        text_content = globals()[method_name](file_url)
    
    logger.success(f"Done load_by_langchain")
    return text_content


# ----------------------------------------------------
# load 후 markdown으로 파일 형식 통일
# ----------------------------------------------------

def _convert_to_markdown(docs: List[Document], category: str, file_url: str) -> str:
    """
    다양한 형태의 Document 리스트를 하나의 마크다운 텍스트로 병합합니다.
    """
    logger.info(f"Converting {category} elements to unified markdown.")
    
    # 문서 시작 부분에 출처 명시 (Markdown H1)
    md_contents = []
    
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
            
        # Unstructured의 category 메타데이터를 활용해 마크다운 구조화
        element_type = doc.metadata.get("category", "")
        
        if element_type == "Title":
            md_contents.append(f"## {content}")
        elif element_type in ["Table", "DataTable"]:
            # 표의 경우 앞뒤로 줄바꿈을 주어 마크다운 구조 유지
            md_contents.append(f"\n{content}\n")
        elif element_type == "ListItem":
            md_contents.append(f"* {content}")
        else:
            md_contents.append(content)

    full_md = "\n\n".join(md_contents)
    return full_md


# ----------------------------------------------------
# 확장자별 추출 함수들
# ----------------------------------------------------

def extract_pdf(file_url: str) -> str:
    """
    1. pymupdf4llm을 사용하여 텍스트 및 표 추출 (일반 텍스트 포함)
    2. fitz를 사용하여 이미지 추출 및 LLM 분석
    3. 페이지별로 텍스트와 이미지 설명 결합
    반환: 마크다운 텍스트 문자열
    """ 
    logger.info(f"Extracting PDF via PyMuPDF4LLM and fitz: {file_url}")
    
    try:
        # 1. PyMuPDF4LLM으로 PDF 전체 콘텐츠를 마크다운으로 변환
        # (일반 텍스트, 제목, 리스트, 테이블 등 모든 텍스트 요소 포함)
        md_docs = pymupdf4llm.to_markdown(file_url, page_chunks=True, write_images=False)

        # 2. 이미지 추출을 위해 fitz로 문서 열기
        doc = fitz.open(file_url)
        full_md = ""
        
        for page_idx, page in enumerate(doc): 
            page_md_content = md_docs[page_idx]["text"]
            image_descriptions = []  # 이미지 설명
            
            # 3. 이미지 추출
            page_dict = page.get_text("dict")
            blocks = page_dict["blocks"]
            for b_idx, block in enumerate(blocks):
                if block["type"] == 1:  # 이미지 블록
                    try:
                        image_bytes = block["image"]
                        # LLM으로 이미지 분석 (bytes 전달)
                        ocr_result = _detect_Image_with_llm(image_bytes)
                        category = ocr_result.get("category", "image")
                        content = ocr_result.get("content", "")
                        image_descriptions.append(f"\n\n### [Category: {category}]\n{content}\n")
                    except Exception as img_err:
                        logger.error(f"Error processing image on page {page_idx}, block {b_idx}: {img_err}")
                        image_descriptions.append(f"\n\n### [Image Analysis Error]\n")
            
            # 4. 텍스트(마크다운)와 이미지 설명 결합
            final_content = page_md_content + "".join(image_descriptions)
            full_md += final_content
        
        doc.close()
        logger.info(f"full_md : {len(full_md)}")
        
        return full_md
        
    except Exception as e:
        logger.error(f"Error in extract_pdf extraction: {e}")
        return ""

def extract_image(file_url: str) -> str:
    """이미지 파일(jpg, png, jpeg) 통합 처리"""
    logger.info(f"Extracting Image via LLM: {file_url}")
    
    try:
        with open(file_url, "rb") as f:
            image_bytes = f.read()
        
        # LLM으로 이미지 분석
        ocr_result = _detect_Image_with_llm(image_bytes)
        category = ocr_result.get("category", "image")
        content = ocr_result.get("content", "")
        
        # 텍스트 생성
        full_content = f"## Analysis Result\n\n### [Category: {category}]\n{content}"
        return full_content
        
    except Exception as e:
        logger.error(f"Error in extract_image: {e}")
        return ""

def extract_md(file_url: str) -> str:
    # Markdown은 이미 형식이 있으므로 single로 가져와서 출처만 붙여줍니다.
    loader = UnstructuredMarkdownLoader(file_url, mode="single")
    return _convert_to_markdown(loader.load(), "Markdown", file_url)

def extract_markdown(file_url: str) -> str:
    return extract_md(file_url)

def extract_docx(file_url: str) -> str:
    loader = UnstructuredWordDocumentLoader(file_url, mode="elements", strategy="fast")
    return _convert_to_markdown(loader.load(), "DOCX", file_url)

def extract_xlsx(file_url: str) -> str:
    loader = UnstructuredExcelLoader(file_url, mode="elements")
    return _convert_to_markdown(loader.load(), "XLSX", file_url)

def extract_xls(file_url: str) -> str:
    return extract_xlsx(file_url)

def extract_csv(file_url: str) -> str:
    # CSVLoader는 기본적으로 Document 리스트를 반환하므로 컨버터 적용 가능
    loader = CSVLoader(file_url)
    return _convert_to_markdown(loader.load(), "CSV", file_url)

def extract_txt(file_url: str) -> str:
    loader = UTF8TextLoader(file_url)
    return _convert_to_markdown(loader.load(), "TXT", file_url)


# ----------------------------------------------------
# 헬퍼 함수
# ----------------------------------------------------

def _detect_Image_with_llm(image_bytes: bytes) -> dict:
    """이미지 바이너리를 받아 Base64로 LLM에 요청 (저장 안함)"""
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        llm = get_llm("gpt-4o-mini")
        detect_image_prompt = get_prompt("detect_image.txt")
        detect_chain = detect_image_prompt | llm
        
        response = detect_chain.invoke({"image": base64_image})
        result = clean_llm_json(response.content)
        
        # 결과가 딕셔너리가 아니면 기본 구조로 변환
        if not isinstance(result, dict):
            return {"category": "image", "content": str(result)}
        
        return result
    
    except Exception as e:
        logger.error(f"LLM Image analysis failed: {e}")
        return {"category": "error", "content": "[Image analysis error]"}