import config.setting
import embed.langchain_chroma_embed as chroma_embed
from etl.langchain_loaders import load_data, save_knowledgebase
from spliter.langchain_chunks import chunk_data
from prompts.load_prompt import get_prompt
from llm.models import get_llm
from langchain_core.output_parsers import StrOutputParser

# 확장자 추출
def get_ext(file_url): 
    return "." + file_url.lower().split(".")[-1]

# 프롬프트
def load_prompt_from_file(path): 
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# 청크데이터 
def get_chunked_docs(file_url):
    ext = get_ext(file_url)
    docs = load_data(file_url, ext)
    chunked_docs = chunk_data(docs, ext)
    return chunked_docs


# vectordb set
def set_vectordb(file_url):
    try:
        chunked_docs = get_chunked_docs(file_url)
        vectordb_id = chroma_embed.build_or_update_vectordb(chunked_docs)
        save_knowledgebase(file_url, chunked_docs, vectordb_id)
        return True
    
    except Exception as e:
        print("set_vectordb Error:", e)
        return False


# 임베딩DB(in memorry) 
def get_vectordb(file_url):
    try:
        chunked_docs = get_chunked_docs(file_url)
        temp_vectordb = chroma_embed.get_temp_vectordb(chunked_docs)
        return True, temp_vectordb
    except Exception as e:
        print("get_vectordb Error:", e)
        return False, None
    

# chain 생성 및 답변 전달(관리자)
def get_llm_response_temp(vectordb, query, history):
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

    return llm_response


# chain 생성 및 답변 전달
def get_llm_response(query, history):
    retriever = chroma_embed.get_retriever(query)

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

    return llm_response



