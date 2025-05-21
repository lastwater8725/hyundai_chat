# 현대자동차 메뉴얼 기반 RAG QA 시스템

현대자동차 PDF 매뉴얼을 기반으로 사용자의 차량 관련 질문에 답변하는 RAG(Retrieval-Augmented Generation) 기반 질의응답 시스템입니다.Streamlit 기반 UI와 FastAPI 백엔드를 분리한 멀티 컨테이너 구조로 설계되었으며, FAISS 벡터 검색 및 EXAONE-3.5 LLM을 활용한 고품질 응답 생성을 지원합니다.또한 HyperCLOVA X Vision 기반 이미지 이해 기능을 추가하여 이미지 기반 질의도 지원합니다.

---

## ✅ 주요 기능

- 현대차 PDF 문서를 구조화 파싱 (pdfminer 기반)
- 텍스트 블록 → 청크화 → 차량 모델 메타정보 추가
- BGE-M3 임베딩 모델로 벡터화 → FAISS 인덱스 구축
- 차량 모델(metadata) 기준으로 FAISS 문서 필터링 지원
- HyperCLOVA X Vision 기반 이미지 분석 → 질문 생성
- LangChain 기반 Retriever + EXAONE LLM 응답 생성
- Streamlit UI ↔ FastAPI 백엔드 연동
- 차량 모델 기준 응답 우선순위 및 출처 인용 지원

사용자
├─ 질문 입력 → Streamlit → FastAPI → FAISS 검색 → ✅ EXAONE 응답 생성
└─ 이미지 업로드 → FastAPI → HyperCLOVA X Vision → 생성된 질문 → 위와 동일 처리


---

## 🛠️ 기술 스택

- Python 3.10
- Streamlit 1.45 (프론트엔드)
- FastAPI + Uvicorn (백엔드)
- LangChain + FAISS
- EXAONE-3.5 / BGE-M3 / HyperCLOVA X Vision
- Docker, Docker Compose

---

## 📁 디렉토리 구조
```
car_manuel/
├── README.md                  # 프로젝트 설명서
├── project.toml              # 프로젝트 메타 정보
├── requirements.txt          # 전체 의존성 패키지 목록
├── docker-compose.yml        # 전체 서비스 정의 (Streamlit + FastAPI)
│
├── data/                     # 데이터 및 파싱 결과
│   ├── car_manual_data/      # 원본 PDF 매뉴얼
│   └── parsed/              # 파싱 및 임베딩 결과
│       └── pdfminer/
│           ├── ...          # 파싱된 JSON
│           ├── chunker/     # 청크 단위 분할 데이터
│           │   ├── *_chunks.json
│           ├── embedding/
│           │   └── faiss_index/
│           │       ├── index.faiss
│           │       └── index.pkl
│
├── src/                      # 소스코드
│   ├── back/                 # 백엔드 (FastAPI 기반)
│   │   ├── Dockerfile
│   │   ├── main.py           # FastAPI 서버 실행 진입점
│   │   ├── app/              # 백엔드 핵심 모듈
│   │   │   ├── api.py        # /query 엔드포인트 처리
│   │   │   ├── modules.py    # LLM, Embedding, FAISS 로더
│   │   │   └── hyperclova_client.py # VLM 분석용 클라이언트
│   │   ├── parser/           # 파서 관련 모듈
│   │   │   ├── pdfminer_parser.py
│   │   │   ├── chunker.py
│   │   │   └── chunker_diff.py
│   │   └── retriever/        # 벡터 검색 관련
│   │       └── embedding.py
│
│   └── front/                # 프론트엔드 (Streamlit)
│       ├── Dockerfile
│       └── main.py           # Streamlit UI 실행

```

## ▶ 실행 방법

### 1. Docker 빌드 및 실행

```bash
docker-compose build
docker-compose up
```

### 2. 접속 경로
```bash
Streamlit UI: http://localhost:8501

FastAPI Docs: http://localhost:8000/docs
```
---

## 🧪 API 사용 예시 (백엔드)

### 🔹 POST/query
```json
{
  "query": "싼타페의 시동이 안 걸릴 때 조치 방법은?",
  "model": "싼타페"
}
```

```json
{
  "answer": "싼타페 시동 문제는 다음과 같은 사항을 점검해야 합니다...",
  "sources": [
    { "page": 11, "model": "싼타페" },
    { "page": 12, "model": "투싼" }
  ]
}
```
### 🔹 POST/generate-question (이미지 기반 질문 생성)

요청:

파일 업로드: multipart/form-data

응답 예시:
```json
{
  "question": "계기판 경고등이 무엇을 의미하나요?"
}
```
---
## 📌 참고사항
지원 차량: 아반떼, 산타페, 캐스퍼, 스타리아, 투싼, 그랜저, 소나타

EXAONE 모델 로딩 시 최초 지연 있음

PDF 구조에 따라 일부 차종은 수동 조정 필요

파싱 및 인덱싱은 최초 1회만 수행 (결과는 /data/에 저장됨)

GPU VRAM 8GB 이상 권장 (EXAONE + BGE-M3 처리 시)
---
👨‍💻 프로젝트 정보
2025 캡스톤디자인 / 최종수