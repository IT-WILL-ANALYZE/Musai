from loguru import logger
import config.setting
import embedders.langchain_chroma_embed as chroma_embed
from etl.langchain_loaders import load_by_langchain, get_ext_from_filename
from etl.unstructured_loaders import load_by_unstructured
from etl.langchain_stores import store_knowledgebase
from chunkers.langchain_chunkers import chunk_structured_by_llm, chunk_data
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.output_parsers import StrOutputParser


# 청크
def get_chunked_docs(file_url, file_structured: dict):
    try:
        logger.info(
            f"Start get_chunked_docs : {file_url, file_structured}"
        )
        
        # 1. load
        text_docs, structured_docs = load_by_langchain(
            file_url,
            mode=file_structured["mode"],
            strategy=file_structured["strategy"]
        )
        
        chunked_docs = []
        # 2. text 형식 chunking
        if text_docs :
            text_chunks = chunk_data(text_docs, get_ext_from_filename(file_url))
            chunked_docs.extend(text_chunks)
            logger.debug(f"chunk_data [text_chunks] : {text_chunks}")
        
        # 3-1. LLM 모델에게 구조화 요청
        if structured_docs :
            structured_chunks = chunk_structured_by_llm(structured_docs)
            chunked_docs.extend(structured_chunks)
            logger.debug(f"chunk_data [structured_chunks] : {chunked_docs}")
        
        logger.success(f"Done get_chunked_docs : {len(chunked_docs)}")
        return chunked_docs
    except Exception as e:
        logger.exception(f"Failed get_chunked_docs : {e}")
        raise
        

# vectordb set
def set_vectordb(file_url, file_structured):
    try:
        logger.info(f"Start set_vectordb : {file_url, file_structured}")

        chunked_docs = get_chunked_docs(file_url, file_structured)
        vectordb_id = chroma_embed.build_or_update_vectordb(chunked_docs)

        store_path = store_knowledgebase(file_url, chunked_docs, vectordb_id)
        logger.success(f"Done set_vectordb : {store_path}")
        return True

    except Exception as e:
        logger.exception(f"Failed set_vectordb : {e}")
        return False


# 임베딩DB(in memory) 
def get_vectordb(file_url, file_structured):
    try:
        logger.info(f"Start get_vectordb : {file_url, file_structured}")

        chunked_docs = get_chunked_docs(file_url, file_structured)
        temp_vectordb = chroma_embed.get_temp_vectordb(chunked_docs)

        logger.success(f"Done get_vectordb : {temp_vectordb}")
        return True, temp_vectordb

    except Exception as e:
        logger.exception(f"Failed get_vectordb : {e}")
        return False, None
    

# chain 생성 및 답변 전달(관리자)
def get_llm_response_temp(vectordb, query, history):
    try:
        logger.info(f"Start get_llm_response_temp : {query, history}")

        retriever = chroma_embed.get_retriever_from_temp(vectordb)
        llm = get_llm("gpt-4.1-mini")

        rag_chain = (
            {
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
                "context": lambda x: retriever.invoke(x["question"])
            }
            | get_prompt("rag_test.txt")
            | llm
            | StrOutputParser()
        )

        llm_response = rag_chain.stream({"question": query, "history": history})
        logger.success(f"Done get_llm_response_temp : {llm_response}")
        return llm_response
    
    except Exception as e:
        logger.exception(f"Failed get_llm_response_temp : {e}")
        return llm_response


# chain 생성 및 답변 전달
def get_llm_response(query, history):
    try:
        logger.info(f"Start get_llm_response : {query, history}")

        retriever = chroma_embed.get_retriever(query)
        llm = get_llm("gpt-4.1-mini")

        rag_chain = (
            {
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
                "context": lambda x: retriever.invoke(x["question"])
            }
            | get_prompt("qa_assistent.txt")
            | llm
            | StrOutputParser()
        )

        llm_response = rag_chain.stream({"question": query, "history": history})
        logger.success(f"Done get_llm_response : {llm_response}")
        return llm_response
    except Exception as e:
        logger.exception(f"Failed get_llm_response : {e}")
        return llm_response
