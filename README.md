# MusAi

패션 데이터 분석을 위한 RAG(Retrieval-Augmented Generation) 기반 챗봇 애플리케이션

## 프로젝트 개요

MusAi는 무신사 데이터를 기반으로 최신 트렌드 및 패션 정보를 제공하는 AI 챗봇입니다. LangChain과 OpenAI를 활용하여 문서 기반 질의응답 시스템을 구현했습니다.

## 주요 기능

- 📄 다양한 문서 형식 지원 (PDF, DOCX, TXT, MD, CSV, XLSX 등)
- 🧩 자동 문서 청킹 및 임베딩
- 🔍 벡터 데이터베이스 기반 유사도 검색
- 💬 대화형 챗봇 인터페이스
- 🔐 관리자 모드 (문서 업로드 및 테스트)

## 배포 방법

### 1. Python 버전 확인

Python 3.8 이상이 필요합니다.

```bash
python --version
# 또는
python3 --version
```

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 OpenAI API 키를 설정하세요.

```env
OPENAI_API_KEY=your_api_key_here
```

### 5. 애플리케이션 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 열리며, 기본 주소는 `http://localhost:8501`입니다.

## 프로젝트 구조

```
Musai/
├── app.py                      # Streamlit 메인 애플리케이션
├── rag_chain.py                # RAG 체인 로직
├── config/
│   └── setting.py              # 환경 설정
├── embed/
│   └── langchain_chroma_embed.py  # 임베딩 및 벡터 DB
├── etl/
│   └── langchain_loaders.py    # 문서 로더
├── spliter/
│   └── langchain_chunks.py     # 문서 청킹
├── llm/
│   └── models.py               # LLM 모델 설정
├── prompts/
│   └── load_prompt.py          # 프롬프트 로더
└── rag_resources/              # 데이터 저장소
    ├── knowledge-base/          # 청킹된 문서 메타데이터
    ├── uploads/                 # 업로드된 원본 파일
    └── vectordb/                # 벡터 데이터베이스
```

## 사용 방법

### 일반 사용자

1. 애플리케이션 실행 후 질문 입력
2. 예시 질문 선택 또는 직접 질문 입력
3. AI가 문서 기반으로 답변 생성

### 관리자 모드

1. 하단의 "관리자 모드" 버튼 클릭
2. 관리자 코드 입력 (기본: `musai123`)
3. 파일 업로드 및 테스트
4. 청킹 및 임베딩 실행
5. 결과 확인 후 저장

## 지원 파일 형식

- 텍스트: `.txt`, `.md`, `.markdown`
- 문서: `.pdf`, `.docx`
- 스프레드시트: `.csv`, `.xlsx`, `.xls`

## 기술 스택

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4
- **Embedding**: OpenAI text-embedding-3-large
- **Vector DB**: ChromaDB
- **Framework**: LangChain

## 주의사항

- `.env` 파일은 Git에 커밋하지 마세요 (보안)
- `rag_resources/` 하위 폴더는 Git에서 제외됩니다
- 대용량 파일 업로드 시 처리 시간이 소요될 수 있습니다

## 문제 해결

### 가상환경 활성화 오류 (Windows)

PowerShell에서 실행 정책 오류가 발생하는 경우:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 패키지 설치 오류

의존성 충돌이 발생하는 경우:

```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

### OpenAI API 키 오류

`.env` 파일이 올바른 위치에 있고, API 키가 정확한지 확인하세요.

## 라이선스

이 프로젝트는 내부 사용을 위한 것입니다.

