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
def get_temp_vectordb(chunked_docs):
    logger.info(f"Start get_temp_vectordb : docs={len(chunked_docs)}")
    try:
        chunked_docs = filter_complex_metadata(chunked_docs)
        vectordb = Chroma.from_documents(chunked_docs, embedding)
        logger.success(f"Done get_temp_vectordb")
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

        logger.success(f"Done build_or_update_vectordb : [vector_ids]={len(vector_ids)}")
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
        logger.success(f"Done get_retriever_from_temp : [retriever]={retriever}")
        return retriever
    except Exception as e:
        logger.exception(f"Failed get_retriever_from_temp : {e}")
        raise


# --------------------------------------------------------
# Persistent vectordb에서 retriever 생성
# --------------------------------------------------------
def get_retriever(query: str):
    logger.info(f"Start get_retriever : [query]='{query}'")

    try:
        vectordb = Chroma(
            persist_directory=VECTORDB_DIR,
            embedding_function=embedding
        )

        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5},
            search_type="similarity"
        )

        logger.success(f"Done get_retriever : [retriever]={retriever}")
        return retriever

    except Exception as e:
        logger.exception(f"Failed get_retriever : {e}")
        raise
