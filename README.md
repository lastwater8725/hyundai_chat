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
 


### Task: DocLayout-YOLO SDK 기반 문서 구조 분석
- **DocLayout-YOLO 공식 SDK**를 활용해 문서 이미지 내의 구조적 요소(텍스트, 표, 제목 등)를 탐지합니다.
- 각 박스마다 **easyOCR**로 텍스트를 추출합니다.
- 결과를 **JSON** 형식으로 저장하여 이후 chunking 및 embedding에 활용할 수 있도록 구성합니다.

---

## 🛠️ 기술 스택
- Python 3.10
- Streamlit (웹 UI)
- LangChain, FAISS (RAG 기반 검색 QA)
- OpenAI API / BGE-M3 (LLM 응답 생성)
- easyOCR, PyMuPDF, pdfminer, DocLayout-YOLO (문서 파싱)

---

## 📂 프로젝트 구조
```
pdf-chatbot-project/
├── app/                     # Streamlit 앱
│   └── main.py
│
├── core/                    # 핵심 기능 (파싱, QA, 벡터화 등)      
│   ├── doclayout_parser.py  # ✅ DocLayout-YOLO SDK 기반 파서, PDF->이미지
│   ├── chunker.py           # 🔜 문단 단위 텍스트 추출 예정
│   ├── qa_generator.py
│   ├── vector_store.py      # FAISS 임베딩 저장
│   └── chatbot.py
│
├── data/                    # 데이터 파일
│   ├── pdfs/                # 원본 PDF
│   ├── images/              # 이미지 파일
│   ├── parsed/              # ✅ 파싱 결과 (JSON)
│   ├── chunks/              # 🔜 문단 단위 조각
│   ├── datasets/            # QA 학습 데이터셋
│   └── vector_store/        # 벡터 인덱스 저장
│
├── models/                  # 모델 설정
│   └── bge_model.py
│
├── config/                  # 환경 설정
│   └── settings.py
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── run.py                   # 전체 파이프라인 실행 스크립트
