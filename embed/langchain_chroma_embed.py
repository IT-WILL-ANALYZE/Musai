from loguru import logger
import config.setting
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

VECTORDB_DIR = "rag_resources/vectordb/chroma_store"

embedding = OpenAIEmbeddings(model="text-embedding-3-large")


# --------------------------------------------------------
# 임시 vectordb 생성 (메모리 기반)
# --------------------------------------------------------
def get_temp_vectordb(chunked_data):
    logger.info(f"Creating TEMP vectordb (docs={len(chunked_data)})")
    try:
        vectordb = Chroma.from_documents(chunked_data, embedding)
        logger.success("TEMP vectordb created successfully")
        return vectordb
    except Exception:
        logger.exception("Failed to create temporary vectordb")
        raise


# --------------------------------------------------------
# 영구 vectordb 생성 또는 업데이트
# --------------------------------------------------------
def build_or_update_vectordb(chunked_docs):
    logger.info(
        f"Building/Updating persistent vectordb: {VECTORDB_DIR} | docs={len(chunked_docs)}"
    )

    try:
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=VECTORDB_DIR
        )

        # add_documents는 자동 저장
        vector_ids = vectordb.add_documents(chunked_docs)

        logger.success(f"Persistent vectordb update complete | added_vectors={len(vector_ids)}")

        return vector_ids
    except Exception:
        logger.exception("Failed to build/update persistent vectordb")
        raise


# --------------------------------------------------------
# TEMP vectordb에서 retriever 생성
# --------------------------------------------------------
def get_retriever_from_temp(vectordb):
    logger.debug("Creating retriever from TEMP vectordb")
    try:
        return vectordb.as_retriever(
            search_kwargs={"k": 5},
            search_type="similarity"
        )
    except Exception:
        logger.exception("Failed to create retriever from TEMP vectordb")
        raise


# --------------------------------------------------------
# Persistent vectordb에서 retriever 생성
# --------------------------------------------------------
def get_retriever(query: str):
    logger.info(f"Loading persistent vectordb and creating retriever | query='{query}'")

    try:
        vectordb = Chroma(
            persist_directory=VECTORDB_DIR,
            embedding_function=embedding
        )

        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5},
            search_type="similarity"
        )

        logger.debug("Retriever created successfully from persistent vectordb")
        return retriever

    except Exception:
        logger.exception("Failed to load persistent vectordb or create retriever")
        raise
