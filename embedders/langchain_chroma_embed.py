import json
import os
from datetime import datetime
from loguru import logger
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 상수 설정
VECTORDB_DIR = "rag_resources/vectordb/chroma_store"
KNOWLEDGE_BASE_PATH = "rag_resources/knowledge-base"

embedding = OpenAIEmbeddings(model="text-embedding-3-large")


# --------------------------------------------------------
# 유틸리티 함수 (날짜 로딩 및 가중치 계산)
# --------------------------------------------------------
def _load_vector_id_to_created_at() -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        if not os.path.isdir(KNOWLEDGE_BASE_PATH):
            return mapping
        for fname in os.listdir(KNOWLEDGE_BASE_PATH):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(KNOWLEDGE_BASE_PATH, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            created_at = data.get("meta", {}).get("created_at", "")
            if not created_at:
                continue
            for chunk in data.get("chunks", []):
                vid = chunk.get("vector_id")
                if vid:
                    mapping[vid] = created_at
    except Exception as e:
        logger.warning(f"Failed to load date mapping: {e}")
    return mapping

def _date_weight(created_at_str: str, decay_per_day: float = 0.01) -> float:
    if not created_at_str:
        return 0.5
    try:
        created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        if created.tzinfo:
            created = created.replace(tzinfo=None)
    except (ValueError, TypeError):
        return 0.5
    days_old = max(0, (datetime.now() - created).days)
    return (1.0 - decay_per_day) ** days_old


# --------------------------------------------------------
# Chroma + 날짜 가중치 커스텀 Retriever (FlashRank 대안)
# --------------------------------------------------------
class DateWeightedChromaRetriever(BaseRetriever):
    vectorstore: Chroma
    search_k: int = 10  # 후보군 10개
    top_n: int = 5     # 최종 반환 5개
    date_decay_per_day: float = 0.01 # 0.01~0.05 : 최신성 강조 / 0.1~0.2 : 최신성 약조
    date_weight_factor: float = 0.4  # 0.2~0.3 : 유사도 위주 / 0.5 ~ : 최신성 강조

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        logger.info(f"Retrieving: {query} (k={self.search_k}, top_n={self.top_n})")

        # 1. 유사도 검색 (점수 포함)
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=self.search_k
        )
        mapping = _load_vector_id_to_created_at()

        scored: list[tuple[Document, float]] = []
        for doc, distance in docs_and_scores:
            # 2. 유사도 점수 정규화 (유사도 점수 높을수록 1에 가깝게)
            relevance = 1.0 / (1.0 + max(0, distance))
            
            # 3. 날짜 가중치 계산
            vid = getattr(doc, "id", None) or doc.metadata.get("vector_id", "")
            created_at = mapping.get(vid, "")
            dw = _date_weight(created_at, self.date_decay_per_day)
            
            # 4. 결합 점수 계산
            combined = relevance * (1.0 - self.date_weight_factor + self.date_weight_factor * dw)
            scored.append((doc, combined))

        # 5. 결합 점수 기준 재정렬 후 상위 top_n개만 반환
        scored.sort(key=lambda x: x[1], reverse=True)
        
        final_docs = [d for d, _ in scored[:self.top_n]]
        logger.debug(f"[retrieved_docs]={final_docs}")
        return final_docs


# --------------------------------------------------------
# Persistent vectordb에서 retriever 생성
# --------------------------------------------------------
def get_retriever(query: str):
    """
    1. Chroma 연결 (text-embedding-3-large 사용)
    2. DateWeightedChromaRetriever: 10개 추출 -> 날짜 가중치 적용 -> 상위 5개 반환
    (FlashRank 제거로 메모리 절약)
    """
    logger.info(f"Start get_retriever : [query]='{query}'")

    try:
        vectordb = Chroma(
            persist_directory=VECTORDB_DIR,
            embedding_function=embedding,
        )

        # 리랭커 없이 날짜 가중치 기반 리트리버 반환
        retriever = DateWeightedChromaRetriever(
            vectorstore=vectordb,
            search_k=10,         # 처음 검색할 후보 수
            top_n=3,             # 최종적으로 LLM에게 줄 문서 수
            date_decay_per_day=0.01,
            date_weight_factor=0.4
        )

        return retriever

    except Exception as e:
        logger.exception(f"Failed get_retriever : {e}")
        raise


# --------------------------------------------------------
# 임시 vectordb 생성 (메모리 기반)
# --------------------------------------------------------
def get_temp_vectordb(chunked_docs):
    logger.info(f"Start get_temp_vectordb : docs={len(chunked_docs)}")
    try:
        chunked_docs = filter_complex_metadata(chunked_docs)
        vectordb = Chroma.from_documents(chunked_docs, embedding)
        
        return vectordb

    except Exception as e:
        logger.exception(f"Failed get_temp_vectordb : {e}")
        raise


# --------------------------------------------------------
# vectordb 생성 또는 업데이트
# --------------------------------------------------------
def build_or_update_vectordb(chunked_docs):
    logger.info(f"Start build_or_update_vectordb : [VECTORDB_DIR]={VECTORDB_DIR} [docs]={len(chunked_docs)}")

    try:
        chunked_docs = filter_complex_metadata(chunked_docs)
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=VECTORDB_DIR
        )

        # add_documents는 자동 저장
        vector_ids = vectordb.add_documents(chunked_docs)

        return vector_ids

    except Exception as e:
        logger.exception(f"Failed build_or_update_vectordb : {e}")
        raise


# --------------------------------------------------------
# TEMP vectordb에서 retriever 생성
# --------------------------------------------------------
def get_retriever_from_temp(vectordb):
    logger.debug(f"Start get_retriever_from_temp")
    
    try:
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5},
            search_type="similarity"
        )
        
        return retriever

    except Exception as e:
        logger.exception(f"Failed get_retriever_from_temp : {e}")
        raise


