# PDF Chatbot Project 🧠📄🤖
# 현대자동차 메뉴얼 기반 챗봇

문서 기반 상담형 챗봇을 구축하는 프로젝트입니다.
PDF 문서를 파싱하여 구조화된 데이터로 변환하고, LLM 기반의 검색 및 응답 시스템으로 연결합니다.

---
### 수정사항
- 프로젝트 구조화를 시켜야 합니다. 0
- 프론트엔드, 백엔드, 0
- 엔트리포인트 정의  0
- RAG 성능 개선인데,
- PDF Parser 중하나만해서 텍스트를 추출한다음에 텍스트를 FAISS 임베딩 시켜야 한다. 
- 지금 PDF->이미지로변환->OCR 하는거는 굉장히 비효율적이다. 
 


### Task: DocLayout-YOLO SDK 기반 문서 구조 분석 및 pdfminer 기반 pdf 파싱 후 rag를 이용한 qa챗봇 제작

- ***ocr기준***
- **DocLayout-YOLO 공식 SDK**를 활용해 문서 이미지 내의 구조적 요소(텍스트, 표, 제목 등)를 탐지합니다.
- 각 박스마다 **easyOCR**로 텍스트를 추출합니다.
- 결과를 **JSON** 형식으로 저장하여 이후 chunking 및 embedding에 활용할 수 있도록 구성합니다.

- *** pdfminer기준***
- pdfminer를 이용하여 정보를 추출합니다. 

- 이후 langchain과 연결하여 qa형 챗봇을 진행합니다.
---

## 🛠️ 기술 스택
- Python 3.10
- linux -> (ubuntu22.04)
- anconda 
- Streamlit (웹 UI)
- LangChain, FAISS (RAG 기반 검색 QA)
- OpenAI API / BGE-M3 (LLM 응답 생성)
- pdfminer.six
- easyOCR, PyMuPDF, pdfminer, DocLayout-YOLO (문서 파싱)

---

## 📂 프로젝트 구조
```
- ocr과 pdf를 구분해주세요

├── README.md
└── car_manuel
    ├── README.md
    ├── data
    ├── project.toml
    └── src
        ├── back
        │   ├── main.py             # 엔트리 포인트
        │   ├── parser              # ocr, pdf구분
        │   │   ├── ocr
        │   │   │   ├── __init__.py
        │   │   │   ├── chunker.py
        │   │   │   ├── doclayout_paser_easy.py
        │   │   │   ├── doclayout_paser_paddle.py
        │   │   │   ├── docyolo.py
        │   │   │   └── embbeder.py
        │   │   └── pdf
        │   │       └── pdfminer_parser.py
        │   └── retriever
        │       ├── langchain_index.py
        │       ├── main.py
        │       └── retriever.py
        └── front


# 📘 자동차 매뉴얼 기반 RAG QA 시스템

자동차 매뉴얼 PDF를 기반으로 한 문서 질의응답 시스템입니다.
문서 레이아웃 분석과 OCR을 통해 텍스트를 추출하고, 임베딩과 FAISS를 활용해 유사한 문서를 검색합니다.

---

## ✅ 파이프라인 요약

- ocr 기준
1. **PDF → 이미지 변환**
2. **DocLayout-YOLO + EasyOCR (또는 PaddleOCR), pdfminer** 를 활용한 레이아웃 분석

- pdfminer 기준
1. ***pdfminer를 활용하여 추출

- 이후는 동일합니다.  
3. **청크 추출**: 블록 단위로 텍스트 청크화
4. **BGE 임베딩 + FAISS 벡터 DB 구축**
5. **LangChain 기반 QA 시스템 구축** (LLM: KULLM3)

---

## 🏗️ 실행 순서 (main은 pdfminer입니다. pdfminer사용법만 적어놓겠습니다.)

```bash
# 1. PDF → json변환
python src/back/parser/pdf/pdfminer_parser.py

# 2. 청크 생성
python src/back/parser/pdf/chunker.py

# 3. 벡터 임베딩 생성
python src/embedder.py

# 4. 랭체인에 맞게 임베딩
python retriever/langchain_index.py

# 5. LangChain 기반 질의응답
python retriever/retriever.py
