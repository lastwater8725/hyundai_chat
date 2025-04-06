# PDF Chatbot Project 🧠📄🤖
# 현대자동차 메뉴얼 기반 챗봇

문서 기반 상담형 챗봇을 구축하는 프로젝트입니다.
PDF 문서를 파싱하여 구조화된 데이터로 변환하고, LLM 기반의 검색 및 응답 시스템으로 연결합니다.

---


### Task: DocLayout-YOLO SDK 기반 문서 구조 분석 및 파싱 후 rag를 이용한 qa챗봇 제작
- **DocLayout-YOLO 공식 SDK**를 활용해 문서 이미지 내의 구조적 요소(텍스트, 표, 제목 등)를 탐지합니다.
- 각 박스마다 **easyOCR**로 텍스트를 추출합니다.
- 결과를 **JSON** 형식으로 저장하여 이후 chunking 및 embedding에 활용할 수 있도록 구성합니다.
- 이후 langchain과 연결하여 qa형 챗봇을 진행합니다.
---

## 🛠️ 기술 스택
- Python 3.10
- linux -> (ubuntu22.04)
- anconda 
- Streamlit (웹 UI)
- LangChain, FAISS (RAG 기반 검색 QA)
- OpenAI API / BGE-M3 (LLM 응답 생성)
- easyOCR, PyMuPDF, pdfminer, DocLayout-YOLO (문서 파싱)

---

## 📂 프로젝트 구조
```
car_manuel/
├── data/
│   ├── pdfs/                             # 원본 PDF
│   ├── images/                           # 변환된 이미지들
│   ├── parsed/                           # YOLO + OCR 결과 JSON 파일들
│   ├── chunks/
│   │   └── chunks.jsonl                  # 청크 단위 텍스트
│   └── embeddings/
│       └── faiss_index/                  # FAISS 인덱스 저장 위치
│
├── src/
│   ├── run.py                            # 전체 파이프라인 실행 스크립트
│   ├── doclayout_paser.py                # YOLO + OCR 기반 파싱 코드
│   ├── chunker.py                        # 청크 추출 코드
│   └── embedder.py                       # 임베딩 및 벡터 DB 저장
│
├── rag/
│   ├── qa_chain.py
│   └── retriever.py                      # LangChain 기반 RAG QA 실행 코드
│
├── pyproject.toml                        # 패키지 의존성 정의
└── README.md                             # 전체 파이프라인 설명


# 📘 자동차 매뉴얼 기반 RAG QA 시스템

자동차 매뉴얼 PDF를 기반으로 한 문서 질의응답 시스템입니다.
문서 레이아웃 분석과 OCR을 통해 텍스트를 추출하고, 임베딩과 FAISS를 활용해 유사한 문서를 검색합니다.

---

## ✅ 파이프라인 요약

1. **PDF → 이미지 변환**
2. **DocLayout-YOLO + EasyOCR (또는 PaddleOCR)** 를 활용한 레이아웃 분석
3. **청크 추출**: 블록 단위로 텍스트 청크화
4. **BGE 임베딩 + FAISS 벡터 DB 구축**
5. **LangChain 기반 QA 시스템 구축** (LLM: KULLM3)

---

## 🏗️ 실행 순서

```bash
# 1. PDF → 이미지 변환 + 레이아웃 분석
python src/run.py

# 2. 청크 생성
python src/chunker.py

# 3. 벡터 임베딩 생성
python src/embedder.py

# 4. 랭체인에 맞게 임베딩
python retriever/langchain_index.py

# 5. LangChain 기반 질의응답
python retriever/retriever.py
