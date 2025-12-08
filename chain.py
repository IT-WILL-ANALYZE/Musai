from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic import hub
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini")


# =========================================
# 1. 문서 로드
# =========================================
def load_data(file_url):
    loader = UnstructuredMarkdownLoader(
        file_url,
        mode="elements",
        strategy="fast",
    )
    return loader.load()


# =========================================
# 2. 문서 청크
# =========================================
def chunk_data(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


# =========================================
# 3. 임베딩 & DB 생성
# =========================================
def embed_data(chunks):
    #chroma에 맞는 형식으로 filter 후 임베딩
    chunks = filter_complex_metadata(chunks) 
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma.from_documents(chunks, embedding)
    return vectordb


# =========================================
# 4. RetrievalQA 생성
# =========================================
def chain_retreivalQA(vector_db, prompt):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


# =========================================
# 5. Prompt 생성
# =========================================
def create_prompt():
    return hub.pull("rlm/rag-prompt")


# =========================================
# 6. Query Tuning
# =========================================
def query_tuning(query):
    dictionary = ["강동영 : 99년생", "오현석 : 94년생",
                  "김유경 : 97년생", "김준혁 : 93년생"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고 아래 사전을 참고하여 질문을 재구성하세요.
        변경이 필요 없으면 그대로 반환하세요.

        사전: {dictionary}
        질문: {query}
    """)

    chain = prompt | llm | StrOutputParser()

    tuned = chain.invoke({"dictionary": dictionary, "query": query})
    return tuned


# =========================================
# MAIN
# =========================================
def rag_generate(file):

    # 1) 문서 로드
    documents = load_data(file)

    # 2) 청킹
    chunks = chunk_data(documents)

    # 3) 임베딩
    vectordb = embed_data(chunks)

    # 4) 프롬프트 생성
    prompt = create_prompt()

    # 5) QA 체인 생성
    qa_chain = chain_retreivalQA(vectordb, prompt)

    # 6) 사용자 질문 입력
    query = input("질문해주세요 : ")

    # 7) 질문 튜닝
    tuned_query = query_tuning(query)

    # 8) 실제 RAG 실행
    answer = qa_chain.invoke({"query": tuned_query})

    print("\n=== 답변 ===")
    print(answer["result"])


if __name__ == "__main__":
    main()
