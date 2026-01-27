"""
reset.py - vectordb 및 knowledge-base 초기화 스크립트

실행 방법:
    python rag_resources/basic-data/reset.py
    
기능:
    1. 기존 vectordb 삭제
    2. knowledge-base 및 uploads 폴더 내용 삭제
    3. basic-data의 JSON 파일을 기반으로 vectordb 및 knowledge-base 재생성
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from langchain_core.documents import Document
import embedders.langchain_chroma_embed as chroma_embed
from etl.langchain_stores import store_knowledgebase


# 경로 설정
BASIC_DATA_DIR = Path(__file__).parent
VECTORDB_DIR = project_root / "rag_resources" / "vectordb" / "chroma_store"
KNOWLEDGE_BASE_DIR = project_root / "rag_resources" / "knowledge-base"
UPLOADS_DIR = project_root / "rag_resources" / "uploads"


def delete_directory_contents(directory: Path, keep_gitkeep: bool = True):
    """디렉토리 내용 삭제 (폴더는 유지)"""
    if not directory.exists():
        logger.info(f"디렉토리가 존재하지 않습니다: {directory}")
        return
    
    logger.info(f"삭제 중: {directory}")
    
    for item in directory.iterdir():
        # .gitkeep 파일은 유지
        if keep_gitkeep and item.name == ".gitkeep":
            continue
            
        if item.is_file():
            item.unlink()
            logger.debug(f"파일 삭제: {item.name}")
        elif item.is_dir():
            shutil.rmtree(item)
            logger.debug(f"폴더 삭제: {item.name}")
    
    logger.success(f"삭제 완료: {directory}")


def delete_vectordb():
    """vectordb 디렉토리 삭제"""
    if VECTORDB_DIR.exists():
        logger.info(f"vectordb 삭제 중: {VECTORDB_DIR}")
        shutil.rmtree(VECTORDB_DIR)
        logger.success("vectordb 삭제 완료")
    else:
        logger.info("삭제할 vectordb가 없습니다")


def load_basic_data_json(json_path: Path) -> dict:
    """basic-data JSON 파일 로드"""
    logger.info(f"JSON 로드 중: {json_path.name}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.debug(f"총 청크 수: {data['meta']['total_chunks']}")
    return data


def convert_chunks_to_documents(data: dict) -> list[Document]:
    """JSON 청크를 LangChain Document 객체로 변환"""
    documents = []
    
    for chunk in data["chunks"]:
        # metadata에서 vector_id 제거 (새로 생성될 것)
        metadata = chunk.get("metadata", {})
        if "vector_id" in metadata:
            del metadata["vector_id"]
        
        # Document 객체 생성
        doc = Document(
            page_content=chunk["content"],
            metadata=metadata
        )
        documents.append(doc)
    
    logger.info(f"Document 변환 완료: {len(documents)}개")
    return documents


def process_basic_data_files():
    """basic-data 폴더의 모든 JSON 파일 처리"""
    json_files = list(BASIC_DATA_DIR.glob("*.json"))
    
    if not json_files:
        logger.warning("basic-data 폴더에 JSON 파일이 없습니다")
        return
    
    logger.info(f"처리할 파일 수: {len(json_files)}")
    
    for json_file in json_files:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"파일 처리 시작: {json_file.name}")
            logger.info(f"{'='*60}")
            
            # 1. JSON 데이터 로드
            data = load_basic_data_json(json_file)
            
            # 2. Document 객체로 변환
            documents = convert_chunks_to_documents(data)
            
            # 3. vectordb에 추가
            logger.info("vectordb에 임베딩 추가 중...")
            vector_ids = chroma_embed.build_or_update_vectordb(documents)
            logger.success(f"vectordb 추가 완료: {len(vector_ids)}개 벡터")
            
            # 4. knowledge-base에 저장
            logger.info("knowledge-base 저장 중...")
            # 원본 파일명 사용 (basic-data의 JSON 파일명)
            file_name = json_file.stem  # 확장자 제외
            
            # store_knowledgebase는 file_url을 받지만 여기서는 가상 경로 사용
            virtual_file_path = f"rag_resources/basic-data/{json_file.name}"
            store_path = store_knowledgebase(virtual_file_path, documents, vector_ids)
            logger.success(f"knowledge-base 저장 완료: {store_path}")
            
        except Exception as e:
            logger.exception(f"파일 처리 실패: {json_file.name} - {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.success("모든 파일 처리 완료!")
    logger.info(f"{'='*60}")


def main():
    """메인 실행 함수"""
    logger.info("="*60)
    logger.info("Reset Script 시작")
    logger.info("="*60)
    
    # 1. vectordb 삭제
    logger.info("\n[단계 1/4] vectordb 삭제")
    delete_vectordb()
    
    # 2. knowledge-base 폴더 내용 삭제
    logger.info("\n[단계 2/4] knowledge-base 폴더 내용 삭제")
    delete_directory_contents(KNOWLEDGE_BASE_DIR, keep_gitkeep=True)
    
    # 3. uploads 폴더 내용 삭제
    logger.info("\n[단계 3/4] uploads 폴더 내용 삭제")
    delete_directory_contents(UPLOADS_DIR, keep_gitkeep=True)
    
    # 4. basic-data 기반으로 vectordb 및 knowledge-base 재생성
    logger.info("\n[단계 4/4] basic-data 기반 vectordb 및 knowledge-base 생성")
    process_basic_data_files()
    
    logger.info("\n" + "="*60)
    logger.success("✅ Reset 완료!")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"예상치 못한 오류 발생: {e}")
        sys.exit(1)
