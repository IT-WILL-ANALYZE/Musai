# OOTData

**OOTData**(Outfit Of The Data)는 *데이터 기반의 착장*을 의미하는 무신사 기반 패션·제품 분석 RAG 챗봇입니다.

## 프로젝트 개요

OOTData는 무신사스탠다드 제품 리뷰, 패션 트렌드 보고서, 통계 자료를 기반으로 최신 트렌드와 제품 분석 정보를 제공하는 AI 챗봇입니다. LangChain과 OpenAI를 활용한 문서 기반 질의응답(RAG) 시스템으로, ChromaDB 벡터 검색과 LLM을 결합해 데이터 근거 기반 답변을 생성합니다.

## 주요 기능

- 📄 **다양한 문서 형식 지원** - PDF, DOCX, TXT, MD, CSV, XLSX 등
- 🧩 **자동 문서 청킹** - LLM 기반 스마트 문서 분할
- ✏️ **Chunk 편집 기능** - 청킹 결과 수동 편집 및 관리
- 🔍 **벡터 검색** - ChromaDB 기반 유사도 검색
- 💬 **대화형 인터페이스** - Streamlit 기반 직관적 UI
- 🔐 **관리자 모드** - 문서 업로드, 청킹, 임베딩, 테스트
- 🐳 **Docker 지원** - 간편한 배포 및 실행

## 기술 스택

### Core
- **Frontend**: Streamlit 1.53+
- **Backend**: Python 3.12
- **LLM**: OpenAI GPT-4
- **Embedding**: OpenAI text-embedding-3-large

### LangChain Ecosystem
- **langchain-core**: 핵심 기능
- **langchain-community**: 문서 로더 및 유틸리티
- **langchain-openai**: OpenAI 통합
- **langchain-chroma**: ChromaDB 통합
- **langchain-text-splitters**: 텍스트 분할

### Storage & Processing
- **Vector DB**: ChromaDB
- **Document Processing**: Unstructured, PyMuPDF
- **Logging**: Loguru

## 배포 방법

### 방법 1: Docker 배포 (권장) 🐳

#### 1. 환경 변수 설정

`.env.example`을 복사하여 `.env` 파일 생성:

```bash
# Windows
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

`.env` 파일 편집:

```env
# OpenAI API 키 (필수)
OPENAI_API_KEY=sk-your-actual-api-key-here

# 관리자 코드 (선택사항, 미설정 시 기능 제한)
ADMIN_CODE=your-custom-admin-code
```

#### 2. Docker 실행

```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

#### 3. 접속

브라우저에서 `http://localhost:8501` 접속

---

### 방법 2: Python 직접 실행

#### 1. Python 버전 확인

Python 3.12 이상 필요:

```bash
python --version
```

#### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

#### 3. 의존성 설치

```bash
# pip 업그레이드
pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt
```

#### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일 생성:

```env
OPENAI_API_KEY=sk-your-api-key-here
ADMIN_CODE=your-admin-code
```

#### 5. 애플리케이션 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 열림 (`http://localhost:8501`)

## 프로젝트 구조

```
OOTData/
├── app.py                          # Streamlit 메인 애플리케이션
├── rag_chain.py                    # RAG 체인 로직
│
├── config/                         # 설정
│   ├── setting.py                  # 환경 설정 (.env 로드)
│   └── logger.py                   # 로깅 설정
│
├── embedders/                      # 임베딩 및 벡터 DB
│   └── langchain_chroma_embed.py   # ChromaDB 임베딩
│
├── etl/                            # 문서 처리 파이프라인
│   ├── langchain_loaders.py        # 문서 로더 (PDF, DOCX 등)
│   ├── langchain_parsers.py        # 문서 파싱
│   ├── langchain_stores.py         # 메타데이터 저장
│   └── unstructured_loaders.py     # Unstructured 로더
│
├── chunkers/                       # 문서 청킹
│   └── langchain_chunkers.py       # LLM 기반 스마트 청킹
│
├── llm/                            # LLM 모델
│   └── models.py                   # OpenAI 모델 설정
│
├── prompts/                        # 프롬프트 관리
│   ├── load_prompt.py              # 프롬프트 로더
│   └── *.txt                       # 프롬프트 템플릿
│
├── pages/                          # Streamlit 페이지
│   ├── chunks.py                   # Chunk/Vector DB 관리 페이지
│   └── prompts.py                  # 프롬프트 테스트 페이지
│
├── images/                         # 이미지 리소스
│   └── logo.png
│
├── rag_resources/                  # 데이터 저장소
│   ├── basic-data/                 # 기본 데이터 - 초기 데이터 셋팅(해당 폴더의 READ.ME참조)
│   ├── knowledge-base/             # 청킹된 문서 메타데이터
│   ├── uploads/                    # 업로드된 원본 파일
│   └── vectordb/                   # 벡터 데이터베이스
│
├── logs/                           # 로그 파일
│
├── .streamlit/                     # Streamlit 설정
│   └── config.toml
│
├── Dockerfile                      # Docker 이미지 빌드
├── docker-compose.yml              # Docker Compose 설정
├── .dockerignore                   # Docker 빌드 제외 파일
│
├── requirements.in                 # 핵심 의존성
├── requirements.txt                # 전체 의존성 (pip-compile 생성)
├── .env.example                    # 환경 변수 예시
└── README.md
```

## 사용 방법

### 일반 사용자 모드

1. 애플리케이션 실행 후 질문 입력
2. 예시 질문(OOTData란, 리뷰 반응, 카테고리, 트렌드 등) 선택 또는 직접 질문 입력
3. AI가 knowledge-base 기반으로 RAG 답변 생성

### 관리자 모드

관리자 모드는 `.env` 파일에 `ADMIN_CODE`가 설정되어 있어야 사용할 수 있습니다.

#### 1. 관리자 인증
- 하단의 "관리자 모드" 버튼 클릭
- `.env`에 설정한 관리자 코드 입력

#### 2. 문서 업로드
- 지원 형식: PDF, DOCX, TXT, MD, CSV, XLSX, XLS
- 파일 선택 후 "업로드" 버튼 클릭

#### 3. 문서 처리 (새 데이터 청킹·임베딩)
1. **청킹 실행**: 문서를 의미 단위로 자동 분할 (LLM 기반 스마트 청킹)
2. **Chunk 관리**: 
   - Chunk 내용 편집
   - Chunk 추가/삭제
   - 원본으로 되돌리기
3. **임베딩 실행**: ChromaDB 벡터 DB 생성 (메모리)
4. **vectorDB에 저장**: knowledge-base JSON 저장 + 영구 ChromaDB 반영

#### 4. 추가 관리
- **Chunk 관리 페이지** (`/chunks`): knowledge-base JSON 파일별 Chunk 조회·편집·VectorDB 동기화
- **프롬프트 관리 페이지** (`/prompts`): 프롬프트 템플릿 테스트

## 챗봇 소개 문서 (청킹용)

**[ootd.md](./ootd.md)**는 OOTData 챗봇의 소개·보유 데이터·기능·사용 방법을 담은 문서입니다. 이 문서를 청킹·임베딩하여 knowledge-base에 포함하면, 사용자가 "OOTData가 뭔가요?", "이 챗봇은 뭘 할 수 있어?" 등 질문 시 이 문서 기반으로 답변합니다.

## 지원 파일 형식

| 형식 | 확장자 | 설명 |
|------|--------|------|
| 텍스트 | `.txt`, `.md`, `.markdown` | 일반 텍스트 |
| 문서 | `.pdf`, `.docx` | PDF, Word 문서 |
| 스프레드시트 | `.csv`, `.xlsx`, `.xls` | 엑셀, CSV |

## 의존성 관리

이 프로젝트는 `pip-tools`를 사용하여 의존성을 관리합니다.

### requirements.in 수정 후 업데이트

```bash
# pip-tools 설치
pip install pip-tools

# requirements.txt 재생성
pip-compile requirements.in -o requirements.txt

# 의존성 설치
pip install -r requirements.txt
```

## 환경 변수

| 변수 | 필수 | 설명 | 기본값 |
|------|------|------|--------|
| `OPENAI_API_KEY` | ✅ | OpenAI API 키 | - |
| `ADMIN_CODE` | ❌ | 관리자 인증 코드 | None (관리자 기능 제한) |

## Docker 상세 설정

### Dockerfile 특징
- **멀티 스테이지 빌드**: 최종 이미지 크기 최적화
- **시스템 의존성**: Tesseract OCR, Poppler, libmagic 포함
- **Python 3.12**: 최신 Python 버전
- **헬스체크**: 자동 컨테이너 상태 모니터링

### 볼륨 마운트
```yaml
volumes:
  - ./rag_resources/uploads:/app/rag_resources/uploads
  - ./rag_resources/knowledge-base:/app/rag_resources/knowledge-base
  - ./rag_resources/vectordb:/app/rag_resources/vectordb
  - ./logs:/app/logs
```

### 포트
- **8501**: Streamlit 웹 인터페이스

## 주의사항

- ⚠️ `.env` 파일은 Git에 커밋하지 마세요 (보안)
- ⚠️ `rag_resources/` 하위 폴더는 Git에서 자동 제외됩니다
- ⚠️ 대용량 파일 업로드 시 처리 시간이 소요될 수 있습니다
- ⚠️ OpenAI API 사용량에 따라 비용이 발생합니다

## 문제 해결

### 가상환경 활성화 오류 (Windows)

PowerShell 실행 정책 오류:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Docker 빌드 오류

캐시 없이 재빌드:

```bash
docker-compose build --no-cache
```

### 패키지 설치 오류

```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

### OpenAI API 키 오류

1. `.env` 파일이 프로젝트 루트에 있는지 확인
2. `OPENAI_API_KEY` 형식이 올바른지 확인 (`sk-`로 시작)
3. API 키가 유효하고 사용량 한도가 남아있는지 확인

### ChromaDB 오류

벡터 DB 초기화:

```bash
rm -rf rag_resources/vectordb/chroma_store
```

## 라이선스

이 프로젝트는 내부 사용을 위한 것입니다.

## 개발자

무신사 데이터 분석팀

---

**Version**: 1.0.0  
**Last Updated**: 2026-03-02
