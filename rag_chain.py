from loguru import logger
import config.setting
import embed.langchain_chroma_embed as chroma_embed
from etl.langchain_loaders import load_data, save_knowledgebase, parse_table_json
from spliter.langchain_chunks import chunk_data
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.output_parsers import StrOutputParser


# 확장자 추출
def get_ext(file_url): 
    return "." + file_url.lower().split(".")[-1]


# 청크데이터 
def get_chunked_docs(file_url):
    ext = get_ext(file_url)
    docs = load_data(file_url, ext)
    chunked_docs = chunk_data(docs, ext)

    table_docs = []

    if ext == ".pdf":
        llm = get_llm("gpt-5")

        page_text = "\n".join(d.page_content for d in docs)

        prompt = get_prompt("find_table_from_pdf.txt")
        chain = prompt | llm

        result = chain.invoke({"page_text": page_text})

        table_text = result.content if hasattr(result, "content") else result

        if table_text.strip() != "NO_TABLE":
            table_docs = parse_table_json(
                table_text,
                source=file_url
            )

    return chunked_docs, table_docs



# vectordb set
def set_vectordb(file_url):
    try:
        logger.info(f"Building or updating vectordb for {file_url}")

        chunked_docs = get_chunked_docs(file_url)
        vectordb_id = chroma_embed.build_or_update_vectordb(chunked_docs)

        save_knowledgebase(file_url, chunked_docs, vectordb_id)
        logger.success(f"VectorDB updated successfully: {vectordb_id}")

        return True

    except Exception as e:
        logger.exception(f"set_vectordb Error for file: {file_url}")
        return False


# 임베딩DB(in memory) 
def get_vectordb(file_url):
    try:
        logger.info(f"Building temporary vectordb for {file_url}")

        chunked_docs = get_chunked_docs(file_url)
        temp_vectordb = chroma_embed.get_temp_vectordb(chunked_docs)

        logger.success("Temporary vectordb created successfully")
        return True, temp_vectordb

    except Exception as e:
        logger.exception(f"get_vectordb Error for file: {file_url}")
        return False, None
    

# chain 생성 및 답변 전달(관리자)
def get_llm_response_temp(vectordb, query, history):
    logger.info(f"[TEMP] Query received: {query}")

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

    logger.debug("Executing temp RAG chain...")
    llm_response = rag_chain.stream({"question": query, "history": history})

    return llm_response


# chain 생성 및 답변 전달
def get_llm_response(query, history):
    logger.info(f"Query received: {query}")

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

    logger.debug("Executing RAG chain...")
    llm_response = rag_chain.stream({"question": query, "history": history})

    return llm_response
