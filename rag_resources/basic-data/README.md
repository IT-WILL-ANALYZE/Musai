# Basic Data Reset Script

이 스크립트는 vectordb와 knowledge-base를 초기 상태로 재설정합니다.

## 📋 기능

1. **기존 데이터 삭제**
   - vectordb 전체 삭제
   - knowledge-base 폴더 내 모든 JSON 파일 삭제
   - uploads 폴더 내 모든 파일 삭제

2. **초기 데이터 생성**
   - `basic-data` 폴더의 JSON 파일들을 기반으로
   - vectordb 재생성 (임베딩 생성 및 저장)
   - knowledge-base JSON 파일 재생성

## 🚀 사용 방법

### 1. 프로젝트 루트에서 실행

```bash
python rag_resources/basic-data/reset.py
```

### 2. basic-data 폴더에서 실행

```bash
cd rag_resources/basic-data
python reset.py
```

## 📁 처리 대상 파일

현재 `basic-data` 폴더에 있는 JSON 파일들:
- `report_01_basic_analysis_1.json`
- `report_02_growth_analysis.json`
- `report_03_segment_analysis.json`

## ⚠️ 주의사항

- **실행 시 기존 데이터가 모두 삭제됩니다**
- OpenAI API 키가 설정되어 있어야 합니다 (임베딩 생성용)
- 인터넷 연결이 필요합니다 (OpenAI API 호출)
- 처리 시간은 파일 크기와 청크 수에 따라 달라집니다

## 📊 실행 로그 예시

```
==============================================================
Reset Script 시작
==============================================================

[단계 1/4] vectordb 삭제
✓ vectordb 삭제 완료

[단계 2/4] knowledge-base 폴더 내용 삭제
✓ 삭제 완료: .../knowledge-base

[단계 3/4] uploads 폴더 내용 삭제
✓ 삭제 완료: .../uploads

[단계 4/4] basic-data 기반 vectordb 및 knowledge-base 생성
============================================================
파일 처리 시작: report_01_basic_analysis_1.json
============================================================
✓ Document 변환 완료: 14개
✓ vectordb 추가 완료: 14개 벡터
✓ knowledge-base 저장 완료: .../report_01_basic_analysis_1.json

[... 다른 파일들도 동일하게 처리 ...]

============================================================
✅ Reset 완료!
============================================================
```

## 🔧 커스터마이징

### 새로운 초기 데이터 추가

`basic-data` 폴더에 JSON 파일을 추가하면 자동으로 처리됩니다.

JSON 파일 형식:
```json
{
  "meta": {
    "file_name": "파일명",
    "total_chunks": 10,
    "created_at": "2026-01-27T00:00:00"
  },
  "chunks": [
    {
      "id": "chunk_001",
      "content": "청크 내용",
      "metadata": {
        "source": "원본 파일 경로",
        "category": "Text",
        ...
      }
    }
  ]
}
```

## 🐛 문제 해결

### OpenAI API 키 오류
```bash
# .env 파일 또는 환경변수 확인
OPENAI_API_KEY=sk-...
```

### 경로 오류
- 프로젝트 루트에서 실행하는지 확인
- 상대 경로가 올바른지 확인

### 권한 오류
- 파일/폴더 쓰기 권한 확인
- vectordb 폴더가 다른 프로세스에서 사용 중인지 확인
