from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

def create_vector_db(chunks):
    chunks = filter_complex_metadata(chunks)
    vectordb = Chroma.from_documents(chunks, embedding)
    return vectordb

def get_similary_vector(query: str):
    vector_query = embedding.embed_query(query)
    vectordb = Chroma(
        persist_directory="./chroma_store",
        embedding_function=embedding
    )
    results = vectordb.similarity_search_by_vector(vector_query, k=5)
    docs = [doc.page_content for doc in results]
    return docs
    






