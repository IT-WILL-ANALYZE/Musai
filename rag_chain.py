from loguru import logger
import config.setting
import json
import embed.langchain_chroma_embed as chroma_embed
from etl.langchain_loaders import load_data, save_knowledgebase, parse_structured_json
from spliter.langchain_chunks import chunk_data
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.output_parsers import StrOutputParser


# 확장자 추출
def get_ext(file_url): 
    return "." + file_url.lower().split(".")[-1]


# 청크데이터 
def get_chunked_docs(file_url, file_structured: bool):
    logger.info(
        f"get_chunked_docs: [file_url]={file_url}, [file_structured]={file_structured}"
    )

    ext = get_ext(file_url)
    chunked_docs = []
    extra_docs = []  # table or structured

    # 기본 로드 (항상 fast)
    docs = load_data(
        file_url,
        ext,
        mode="elements",
        strategy="fast"
    )

    # 구조화 옵션이 켜져 있을 때만 LLM 판단
    if ext not in ["xlsx", "xls", "csv"] and file_structured:
        llm = get_llm("gpt-4.1-mini")

        docs_text = "\n".join(
            d.page_content for d in docs if d.page_content
        )

        logger.debug(f"[STRUCTURE CHECK INPUT]\n{docs_text}")

        # -------------------------
        # 1. Table 판단 (엄격)
        # -------------------------
        table_prompt = get_prompt("detect_table_by_llm.txt")
        table_chain = table_prompt | llm

        table_result = table_chain.invoke(
            {"docs_text": docs_text}
        )

        table_text = (
            table_result.content
            if hasattr(table_result, "content")
            else table_result
        )

        logger.info(f"[TABLE DETECTION RESULT] {table_text}")

        if table_text.strip() != "NO_TABLE":
            extra_docs.extend(
                parse_structured_json(
                    table_text,
                    source=file_url
                )
            )
        else:
            # -------------------------
            # 2️. Structured 판단 (관대)
            # -------------------------
            structured_prompt = get_prompt("detect_structured_by_llm.txt")
            structured_chain = structured_prompt | llm

            structured_result = structured_chain.invoke(
                {"docs_text": docs_text}
            )

            structured_text = (
                structured_result.content
                if hasattr(structured_result, "content")
                else structured_result
            )

            logger.info(f"[STRUCTURED DETECTION RESULT] {structured_text}")

            try:
                structured_json = json.loads(structured_text)
            except Exception:
                structured_json = {"decision": "UNSTRUCTURED"}

            if structured_json.get("decision") == "STRUCTURED":
                stype = structured_json["structure_type"]

                if stype == "qa_pairs":
                    extract_prompt = get_prompt("extract_qa_pairs.txt")

                elif stype == "list":
                    extract_prompt = get_prompt("extract_list.txt")

                elif stype == "definition":
                    extract_prompt = get_prompt("extract_definition.txt")

                else:
                    extract_prompt = None

                # 실제 구조화
                if extract_prompt is not None:
                    extract_chain = extract_prompt | llm

                    extract_result = extract_chain.invoke(
                        {"docs_text": docs_text}
                    )

                    extract_text = (
                        extract_result.content
                        if hasattr(extract_result, "content")
                        else extract_result
                    )

                    extra_docs.extend(
                        parse_structured_json(
                            extract_text,
                            source=file_url
                        )
                    )

    # -------------------------
    # chunking (항상 수행)
    # -------------------------
    chunked_docs = chunk_data(docs, ext)

    logger.success(
        f"get_chunked_docs done | chunked={len(chunked_docs)}, extra={len(extra_docs)}"
    )

    return chunked_docs + extra_docs


# vectordb set
def set_vectordb(file_url, file_structured):
    try:
        logger.info(f"Building or updating vectordb for {file_url}")

        chunked_docs = get_chunked_docs(file_url, file_structured)
        vectordb_id = chroma_embed.build_or_update_vectordb(chunked_docs)

        save_knowledgebase(file_url, chunked_docs, vectordb_id)
        logger.success(f"VectorDB updated successfully: {vectordb_id}")

        return True

    except Exception as e:
        logger.exception(f"set_vectordb Error for file: {file_url}")
        return False


# 임베딩DB(in memory) 
def get_vectordb(file_url, file_structured):
    try:
        logger.info(f"Building temporary vectordb for {file_url}")

        chunked_docs = get_chunked_docs(file_url, file_structured)
        temp_vectordb = chroma_embed.get_temp_vectordb(chunked_docs)

        logger.success("Temporary vectordb created successfully")
        return True, temp_vectordb

    except Exception as e:
        logger.exception(f"get_vectordb Error for file: {file_url}")
        return False, None
    

# chain 생성 및 답변 전달(관리자)
def get_llm_response_temp(vectordb, query, history):
    logger.info(f"Query received: {query}")

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
