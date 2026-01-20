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

def load_by_langchain(file_url: str) -> List[Document]:
    logger.info(f"Start load_by_langchain : {file_url}")
    
    # 확장자 추출
    ext = os.path.splitext(file_url)[1].lower().replace('.', '')
    
    # 이미지 통합 처리 (.jpg, .png, .jpeg)
    if ext in ['jpg', 'jpeg', 'png']:
        return extract_image(file_url)
    
    # 동적 함수 매핑 (extract_{ext})
    method_name = f"extract_{ext}"
    if globals().get(method_name):
        return globals()[method_name](file_url)
    else:
        logger.error(f"Unsupported extension: {ext}")
        return []


# ----------------------------------------------------
# load 후 markdown으로 파일 형식 통일
# ----------------------------------------------------

def _convert_to_markdown(docs: List[Document], category: str, file_url: str) -> List[Document]:
    """
    다양한 형태의 Document 리스트를 하나의 마크다운 Document로 병합합니다.
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
    return [Document(page_content=full_md, metadata={"source": file_url, "category": category})]


# ----------------------------------------------------
# 확장자별 추출 함수들
# ----------------------------------------------------

def extract_pdf(file_url: str) -> List[Document]:
    """
    1. pymupdf4llm을 사용하여 텍스트 및 표 추출
    2. fitz를 사용하여 이미지 추출 및 LLM 분석
    3. 순서에 맞게 병합
    """
    logger.info(f"Extracting PDF via PyMuPDF4LLM and fitz: {file_url}")
    
    # 1. PyMuPDF4LLM으로 마크다운 기반 텍스트/표 추출
    md_text = pymupdf4llm.to_markdown(file_url)
    
    # 2. 이미지 추출 및 LLM 분석 (Base64)
    doc = fitz.open(file_url)
    llm_image_descriptions = []
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Base64로 LLM 요청
            description = _detect_structures_with_llm(image_bytes)
            # 마크다운 인용구(>) 형식을 사용하여 이미지 설명임을 명시
            llm_image_descriptions.append(f"\n\n> ### [Image Analysis (Page {page_index+1})]\n> {description}\n")

    # 3. 텍스트와 이미지 분석 내용 병합
    full_content = f"# Source: {os.path.basename(file_url)}\n\n" + md_text + "\n" + "\n".join(llm_image_descriptions)
    
    return [Document(page_content=full_content, metadata={"source": file_url, "category": "PDF"})]

def extract_image(file_url: str) -> List[Document]:
    """이미지 파일(jpg, png, jpeg) 통합 처리"""
    logger.info(f"Extracting Image via LLM: {file_url}")
    with open(file_url, "rb") as f:
        image_bytes = f.read()
    
    description = _detect_structures_with_llm(image_bytes)
    full_content = f"# Source Image: {os.path.basename(file_url)}\n\n## Analysis Result\n{description}"
    return [Document(page_content=full_content, metadata={"source": file_url, "category": "Image"})]

def extract_md(file_url: str) -> List[Document]:
    # Markdown은 이미 형식이 있으므로 single로 가져와서 출처만 붙여줍니다.
    loader = UnstructuredMarkdownLoader(file_url, mode="single")
    return _convert_to_markdown(loader.load(), "Markdown", file_url)

def extract_markdown(file_url: str) -> List[Document]:
    return extract_md(file_url)

def extract_docx(file_url: str) -> List[Document]:
    loader = UnstructuredWordDocumentLoader(file_url, mode="elements", strategy="fast")
    return _convert_to_markdown(loader.load(), "DOCX", file_url)

def extract_xlsx(file_url: str) -> List[Document]:
    loader = UnstructuredExcelLoader(file_url, mode="elements")
    return _convert_to_markdown(loader.load(), "XLSX", file_url)

def extract_xls(file_url: str) -> List[Document]:
    return extract_xlsx(file_url)

def extract_csv(file_url: str) -> List[Document]:
    # CSVLoader는 기본적으로 Document 리스트를 반환하므로 컨버터 적용 가능
    loader = CSVLoader(file_url)
    return _convert_to_markdown(loader.load(), "CSV", file_url)

def extract_txt(file_url: str) -> List[Document]:
    loader = UTF8TextLoader(file_url)
    return _convert_to_markdown(loader.load(), "TXT", file_url)


# ----------------------------------------------------
# 헬퍼 함수
# ----------------------------------------------------

def _detect_structures_with_llm(image_bytes: bytes) -> str:
    """이미지 바이너리를 받아 Base64로 LLM에 요청 (저장 안함)"""
    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        llm = get_llm("gpt-4o-mini")
        detect_structures_prompt = get_prompt("detect_structures.txt")
        detect_chain = detect_structures_prompt | llm
        
        response = detect_chain.invoke({"image": base64_image})
        return clean_llm_json(response.content)
    
    except Exception as e:
        logger.error(f"LLM Image analysis failed: {e}")
        return "[Image analysis error]"