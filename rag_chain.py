from loguru import logger
import config.setting
import embed.langchain_chroma_embed as chroma_embed
from etl.langchain_loaders import load_by_langchain
from etl.unstructured_loaders import load_by_unstructured
from etl.langchain_stores import store_knowledgebase
from spliter.langchain_chunks import chunk_structured_by_llm, chunk_data
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.output_parsers import StrOutputParser


# 확장자 추출
def get_ext(file_url): 
    return "." + file_url.lower().split(".")[-1]

# 청크
def get_chunked_docs(file_url, file_structured: bool):
    try:
        logger.info(
            f"Start get_chunked_docs : {file_url, file_structured}"
        )

        ext = get_ext(file_url)
        
        # 1. 기본 로드
        # docs = load_by_langchain(
        #     file_url,
        #     ext,
        #     mode="elements",
        #     strategy="fast"
        # )
        docs = load_by_unstructured(file_url,ext)

        # 2. normal chunking (항상 수행)
        chunked_docs = chunk_data(docs, ext)
        
        # 3. 구조화(표, 리스트, QA등) 필요한 경우 LLM 모델에게 요청
        if ext not in ["xlsx", "xls", "csv"] and file_structured: 
            chunked_docs = chunked_docs + chunk_structured_by_llm(chunked_docs)
        
        logger.success(f"Done get_chunked_docs : {len(chunked_docs)}")
    except Exception as e:
        logger.exception(f"Failed get_chunked_docs : {e}")

        return chunked_docs


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
