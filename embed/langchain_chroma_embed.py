import config.setting
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

VECTORDB_DIR = "rag_resources/vectordb/chroma_store"

embedding = OpenAIEmbeddings(model="text-embedding-3-large")


def get_temp_vectordb(chunked_data):
    vectordb = Chroma.from_documents(chunked_data, embedding)
    return vectordb


def build_or_update_vectordb(chunked_docs):
    
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=VECTORDB_DIR
    )

    # add_documents()는 자동 저장됨
    vector_ids = vectordb.add_documents(chunked_docs)

    return vector_ids


def get_retriever_from_temp(vectordb):
    return vectordb.as_retriever(search_kwargs={"k": 5}, search_type="similarity")


def get_retriever(query: str):
    vectordb = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embedding
    )

    return vectordb.as_retriever(search_kwargs={"k": 5}, search_type="similarity")

    






