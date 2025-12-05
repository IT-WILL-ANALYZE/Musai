from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def embed_data(chunks):
    #chroma에 맞는 형식으로 filter 후 임베딩
    chunks = filter_complex_metadata(chunks) 
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma.from_documents(chunks, embedding)
    return vectordb





