import json
import config.setting
from loguru import logger
import embedders.langchain_chroma_embed as chroma_embed
from etl.langchain_loaders import load_by_langchain
from etl.langchain_parsers import clean_llm_json
from etl.langchain_stores import store_knowledgebase
from chunkers.langchain_chunkers import chunk_format_md
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.output_parsers import StrOutputParser


def _limit_history(history, max_messages: int = 5) -> str:
    if history is None:
        return ""

    if isinstance(history, list):
        msgs = history[-max_messages:]
        lines: list[str] = []
        for m in msgs:
            if isinstance(m, dict):
                role = str(m.get("role", "")).strip()
                content = str(m.get("content", "")).strip()
                if role and content:
                    lines.append(f"{role}: {content}")
                elif content:
                    lines.append(content)
            else:
                lines.append(str(m).strip())
        return "\n".join([ln for ln in lines if ln])

    if isinstance(history, str):
        lines = history.splitlines()
        if len(lines) > max_messages:
            lines = lines[-max_messages:]
        return "\n".join(lines).strip()

    return str(history).strip()


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
    

# chain 생성 및 답변 전달
def get_llm_response(question, history, vectordb=None):
    """
    반환: (spinner_text, stream_generator)
    app.py에서 spinner 문구 + st.write_stream(stream) 사용.
    """
    spinner_text = (
        "🧪 OOTData가 테스트 모드로 답변을 생성하고 있어요..."
        if vectordb is not None
        else "🤔 OOTData가 답변을 생성하고 있어요..."
    )

    def _stream():
        try:
            logger.info("Start get_llm_response")
            history_limited = _limit_history(history, max_messages=5)  # 최근 이력 5개만 사용(토큰비용 절약)
            logger.debug(f"[question]='{question}', [history(last5)]='{history_limited}'")
            llm = get_llm("gpt-5.2")
            # 1) 전처리 question 분석 (optimizer는 JSON 배열 문자열 반환)
            optimizer_chain = (
                get_prompt("released_optimizer.txt")
                | llm
                | StrOutputParser()
            )
            optimizer_output = optimizer_chain.invoke(
                {"question": question, "history": history_limited}
            )
            logger.debug(f"[optimizer_output]='{optimizer_output}'")

            try:
                cleaned = clean_llm_json(optimizer_output)
                parsed = json.loads(cleaned)
                query_list = parsed if isinstance(parsed, list) else [question]
            except (json.JSONDecodeError, TypeError):
                query_list = [question]

            # 문자열 query만 정리 (빈 배열이면 검색 스킵)
            query_list = [
                q.strip() for q in query_list
                if isinstance(q, str) and q.strip()
            ]
            query_list = list(dict.fromkeys(query_list))

            # 2) 검색 질의 생성(retrieval) - optimizer 결과가 있을 때만 실행
            retrievals = []
            if query_list:
                retriever = chroma_embed.set_retriever(vectordb)
                for q in query_list:
                    docs = retriever.invoke(q)
                    retrievals.append({"query": q, "context": docs})

            # 3) 프롬프트용 context 문자열 생성
            context_parts = []
            for r in retrievals:
                for doc in r["context"]:
                    context_parts.append(doc.page_content)
            context = "\n\n".join(context_parts) if context_parts else ""
            logger.debug(f"[context_len]={len(context)}")

            # 4) 최종 결과 생성(LLM) — 여기서 대부분의 지연 발생 (gpt-5 호출·TTFT)
            rag_chain = (
                {
                    "question": lambda x: x["question"],
                    "history": lambda x: x["history"],
                    "context": lambda x: x["context"],
                }
                | get_prompt("released_responser.txt")
                | llm
                | StrOutputParser()
            )

            for chunk in rag_chain.stream({"question": question, "history": history_limited, "context": context}):
                yield chunk
        except Exception as e:
            logger.exception(f"Failed get_llm_response : {e}")
            yield ""

    return spinner_text, _stream()


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
                    retriever = chroma_embed.set_retriever()
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
