import json
import os
from datetime import datetime
from loguru import logger
import config.setting
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

VECTORDB_DIR = "rag_resources/vectordb/chroma_store"
KNOWLEDGE_BASE_PATH = "rag_resources/knowledge-base"

embedding = OpenAIEmbeddings(model="text-embedding-3-large")


# --------------------------------------------------------
# JSON 기반 날짜 가중치 매핑
# --------------------------------------------------------
def _load_vector_id_to_created_at() -> dict[str, str]:
    """
    knowledge-base JSON 파일을 분석하여 vector_id -> created_at(ISO) 매핑 반환.
    각 chunk의 vector_id와 상위 meta.created_at을 연결.
    """
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
        logger.debug(f"Loaded date mapping for {len(mapping)} chunks from knowledge-base")
    except Exception as e:
        logger.warning(f"Failed to load date mapping: {e}")
        
    return mapping


def _date_weight(created_at_str: str, decay_per_day: float = 0.01) -> float:
    """
    created_at(ISO 문자열) 기준 날짜 가중치 계산. 최신일수록 1에 가깝고, 오래될수록 감소.
    decay_per_day: 일 단위 감쇠율 (0.01 = 100일 경과 시 약 0.37)
    """
    if not created_at_str:
        return 0.5  # 정보 없으면 중립
    try:
        created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        if created.tzinfo:
            created = created.replace(tzinfo=None)
    except (ValueError, TypeError):
        return 0.5
    days_old = max(0, (datetime.now() - created).days)
    
    return (1.0 - decay_per_day) ** days_old


# --------------------------------------------------------
# Chroma + 날짜 가중치 커스텀 Retriever
# --------------------------------------------------------
class DateWeightedChromaRetriever(BaseRetriever):
    """
    Chroma 유사도 검색 후 JSON 기반 created_at 날짜 가중치를 적용한 Retriever.
    """

    vectorstore: Chroma
    search_k: int = 20
    date_decay_per_day: float = 0.01 # 0~1, 날짜 가중치 감소 비율
    date_weight_factor: float = 0.3  # 0~1, 유사도 대비 날짜 가중치 비중

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        logger.info(f"Start DateWeightedChromaRetriever : [query]='{query}', [search_k]={self.search_k}")

        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query, k=self.search_k
        )
        mapping = _load_vector_id_to_created_at()

        scored: list[tuple[Document, float]] = []
        for doc, distance in docs_and_scores:
            # Chroma distance: 낮을수록 유사. relevance = 1/(1+d) 로 변환
            relevance = 1.0 / (1.0 + max(0, distance))
            # vector_id: Chroma Document.id 또는 metadata
            vid = getattr(doc, "id", None) or doc.metadata.get("vector_id", "")
            created_at = mapping.get(vid, "")
            dw = _date_weight(created_at, self.date_decay_per_day)
            # combined = relevance * (1 - date_factor + date_factor * date_weight)
            combined = relevance * (1.0 - self.date_weight_factor + self.date_weight_factor * dw)
            scored.append((doc, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [d for d, _ in scored]


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
# 영구 vectordb 생성 또는 업데이트
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


# --------------------------------------------------------
# Persistent vectordb에서 retriever 생성
# --------------------------------------------------------
def get_retriever(query: str):
    """
    1. Chroma 유지 (저장 안정성)
    2. JSON knowledge-base 분석 → 날짜 가중치 적용 (DateWeightedChromaRetriever)
    3. FlashRank 리랭킹 (가볍고 빠른 의미적 정밀도)
    retriever 객체 반환 (invoke(question) 호출로 문서 검색)
    """
    logger.info(f"Start get_retriever : [query]='{query}'")

    try:
        # 1. Chroma 연결
        vectordb = Chroma(
            persist_directory=VECTORDB_DIR,
            embedding_function=embedding,
        )

        # 2. Chroma + JSON 날짜 가중치 Retriever (1단계: Recall + 날짜 보정)
        date_weighted_retriever = DateWeightedChromaRetriever(vectorstore=vectordb)

        # 3. FlashRank 리랭커 (2단계: Precision, 한국어 MultiBERT) # ms-marco-MiniLM-L-6-v2
        compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12", top_n=5)

        # 4. 두 단계 결합하여 retriever 반환 (rag_chain에서 invoke 호출)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=date_weighted_retriever,
        )
        logger.info(f"Done get_retriever : {compression_retriever}")

        return compression_retriever

    except Exception as e:
        logger.exception(f"Failed get_retriever : {e}")
        raise