import config.setting
from loguru import logger
import embedders.langchain_chroma_embed as chroma_embed
from etl.langchain_loaders import load_by_langchain
from etl.langchain_stores import store_knowledgebase
from chunkers.langchain_chunkers import chunk_format_md
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.output_parsers import StrOutputParser


# 청크
def get_chunked_docs(file_url):
    try:
        logger.info(
            f"Start get_chunked_docs : {file_url}"
        )
        
        # 1. load
        content_md = load_by_langchain(file_url)

        # 2. chunk
        chunked_docs = chunk_format_md(content_md)
        
        logger.success(f"Done get_chunked_docs : {len(chunked_docs)} chunks created")
        return chunked_docs
    except Exception as e:
        logger.exception(f"Failed get_chunked_docs : {e}")
        raise
        

# vectordb set
def set_vectordb(file_url, chunked_docs):
    try:
        logger.info(f"Start set_vectordb")

        vectordb_id = chroma_embed.build_or_update_vectordb(chunked_docs)
        store_path = store_knowledgebase(file_url, chunked_docs, vectordb_id)
        
        logger.success(f"Done set_vectordb : {store_path}")
        return True

    except Exception as e:
        logger.exception(f"Failed set_vectordb : {e}")
        return False


# 임베딩DB(in memory) 
def get_vectordb(chunked_docs):
    try:
        logger.info(f"Start get_vectordb : {len(chunked_docs)}")

        temp_vectordb = chroma_embed.get_temp_vectordb(chunked_docs)

        logger.success(f"Done get_vectordb : {temp_vectordb}")
        return True, temp_vectordb

    except Exception as e:
        logger.exception(f"Failed get_vectordb : {e}")
        return False, None
    

# chain 생성 및 답변 전달(관리자)
def get_llm_response_temp(vectordb, query, history):
    try:
        logger.info("Start get_llm_response_temp")
        logger.debug(f"[query]='{query}', [history]='{history}'")

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

        # 스트리밍 방식으로 각 토큰을 하나씩 yield
        for chunk in rag_chain.stream({"question": query, "history": history}):
            yield chunk
    
    except Exception as e:
        logger.exception(f"Failed get_llm_response_temp : {e}")
        yield ""


# chain 생성 및 답변 전달
def get_llm_response(query, history):
    try:
        logger.info("Start get_llm_response")
        logger.debug(f"[query]='{query}', [history]='{history}'")

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

        # 스트리밍 방식으로 각 토큰을 하나씩 yield
        for chunk in rag_chain.stream({"question": query, "history": history}):
            yield chunk
    except Exception as e:
        logger.exception(f"Failed get_llm_response : {e}")
        yield ""


# chain 생성 및 답변 전달 (프롬프트 테스트용)
def get_llm_response_test(query, llm_model, prompt_text, variables=None, use_rag_for_context=False):
    """
    프롬프트 테스트용: query, llm_model, prompt_text를 받아 chain 생성 및 스트리밍 답변 전달
    variables: 프롬프트 템플릿 변수 dict (question, content 등)
    use_rag_for_context: True 시 VectorDB에서 context 검색하여 사용
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate

        logger.info("Start get_llm_response_test")
        logger.debug(f"[query]='{query}', [model]='{llm_model}', [use_rag]='{use_rag_for_context}'")

        template = ChatPromptTemplate.from_template(prompt_text)
        input_vars = set(template.input_variables)

        if variables is None:
            variables = {}

        # history, context 자동 설정 (템플릿에 해당 변수가 있는 경우)
        if "history" in input_vars:
            variables = {**variables, "history": ""}
        if "context" in input_vars:
            if use_rag_for_context and query:
                try:
                    retriever = chroma_embed.get_retriever(query)
                    context_docs = retriever.invoke(query)
                    context = "\n\n".join(doc.page_content for doc in context_docs) if context_docs else ""
                except Exception as e:
                    logger.warning(f"VectorDB 검색 실패: {e}")
                    context = ""
            else:
                context = ""
            variables = {**variables, "context": context}

        llm = get_llm(llm_model)
        chain = template | llm | StrOutputParser()

        for chunk in chain.stream(variables):
            yield chunk
    except Exception as e:
        logger.exception(f"Failed get_llm_response_test : {e}")
        yield ""
