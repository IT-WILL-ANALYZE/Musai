📘 RAG Prototype Storyboard

1. 프로젝트 준비 과정
   1.1 GitHub 저장소 클론

사용자는 먼저 GitHub에서 프로젝트를 클론한다.

git clone https://github.com/사용자/your-rag-project.git
cd your-rag-project

1.2 Python 가상환경 생성 및 활성화
python -m venv .venv

Windows:

.venv\Scripts\activate

Mac/Linux:

source .venv/bin/activate

1.3 필수 라이브러리 설치 (requirements.txt)

프로젝트에는 아래 기술 스택이 포함된다:

LangChain

ChromaDB

OpenAI API

Unstructured

dotenv

사용자는 다음 명령으로 모든 의존성을 설치한다:

pip install -r requirements.txt

2. .env 셋팅

OpenAI API Key 및 환경 설정을 위해 루트 디렉토리에 .env 파일 추가:

OPENAI_API_KEY=your_key_here

3. 프로젝트 실행
   python main.py

실행하면 프로그램은 다음 단계로 진행된다.

📚 4. RAG Storyboard Flow

아래는 main.py가 실행되는 전체 흐름을 단계별 스토리보드로 정리한 것이다.

🎬 Step 1 — Markdown 문서 로드

UnstructuredMarkdownLoader

mode="elements"

Markdown 파일을 구조별 요소(Document)로 분해

예: 제목, 문단, 리스트 등 각각 Document로 반환

documents = load_data("chunk_dummy/analyze_kaiba.md")

🎬 Step 2 — 문서 청킹 (Chunking)

RecursiveCharacterTextSplitter

chunk_size=500

overlap=50

elements 기반으로 잘게 나뉜 Document들을 다시 자연스러운 길이로 병합/분리

chunks = chunk_data(documents)

🎬 Step 3 — 임베딩 및 벡터 저장소 구축

OpenAI Embeddings + Chroma Vector DB

"text-embedding-3-large" 사용

Chroma의 metadata 제약을 위해 filter_complex_metadata() 실행
→ list/dict 메타데이터 자동 제거

vectordb = embed_data(chunks)

ChromaDB는 in-memory DB로 생성되며
문서 임베딩 + metadata를 저장한다.

🎬 Step 4 — RetrievalQA 체인 구축

RetrievalQA.from_chain_type

LLM: gpt-4.1-mini

Retriever: Chroma 기반 유사도 검색

Prompt: rlm/rag-prompt (Hub에서 로드)

qa_chain = chain_retreivalQA(vectordb, prompt)

이로써 검색 + 응답 생성이 가능한 RAG 체인이 완성됨.

🎬 Step 5 — 사용자 질문 입력

프로그램은 사용자에게 질문을 입력받는다:

사용자 입력 → query

🎬 Step 6 — Query Tuning (질문 보정)

다음과 같은 방식으로 사전(dictionary) 기반 질문 튜닝:

사전: ["강동영 : 99년생", ...]

사용자의 질문을 사전과 비교

필요 시 질문을 변형해 검색 정확도 향상

tuned_query = query_tuning(query)

🎬 Step 7 — RAG 실행

튜닝된 쿼리를 RetrievalQA 체인에 전달한다:

answer = qa_chain.invoke({"query": tuned_query})

LLM은:

튜닝된 질문을 기반으로

ChromaDB에서 관련 문서를 검색하고

Prompt 기반으로 문맥을 참고한 답변을 생성한다.

🎬 Step 8 — 최종 답변 출력
=== 답변 ===
<LLM이 생성한 최종 응답>

이로써 전체 RAG 파이프라인이 동작한다.

📦 전체 구조 요약 흐름도
Markdown 파일
↓
Unstructured Loader (elements)
↓
Chunking (RecursiveCharacterTextSplitter)
↓
Embedding (OpenAI)
↓
Vector DB (Chroma)
↓
Retriever + LLM 결합 (RetrievalQA)
↓
Query Tuning (사전 기반)
↓
최종 RAG 응답
