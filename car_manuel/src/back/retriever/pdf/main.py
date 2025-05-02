#rag pipline -> langchain + faiss 

import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_huggingface import HuggingFaceBgeEmbeddings  ! bge와 호환 에러인듯 함

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

index_path = './data/parsed/pdfminer/embedding/faiss_index'
embedding_model_name = "BAAI/bge-m3"
llm_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

#모델 로드
embedding = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)

# 벡터 db 로드
db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)

# === [4] LLM 로딩 (EXAONE) ===
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    trust_remote_code=True
).to("cuda")

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=512,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

#프롬프트 템플릿 정의
template = """
너는 현대자동차 매뉴얼 기반 AI야.
아래 문서를 참고해서 사용자의 질문에 친절하고 정확하게 답변해줘.
각 문서는 특정 차량 모델({model})에 대한 내용이야.
현재 문서에는 아반떼, 싼타페, 아이오닉5에 대한 정보가 포함되어있어
관련 내용 우선적으로 참고해서 바탕으로 안내해줘.
참고한 rag문서의 페이지와 차량 이름을 항상 같이 출력해줘.
rag문서 내용을 명확하게 파악하고 답변해줘.
문서 내용과 함께 [출처 정보]도 포함되어 있어. 예시처럼 페이지 번호와 차량 모델을 인용해줘:
예) [page 5, model: 아반떼]
질문한 차량 모델과 문서의 모델명이 일치할 경우 해당 내용을 가장 신뢰도 높은 정보로 간주해서 답변을 구성해줘.
다른 차량 문서에 유사한 정보가 있을 수 있으니 필요시 참고하고 어떤 문서도 함께 참고했는지 명시해줘

문서 내용:
{context}

질문:
{question}

답변을 문장으로 완성해줘.
답변:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "query", "model"],
    template=template.replace("{question}", "{query}")
)

# langchain rag qa 생성
llm_chain = LLMChain(llm=llm, prompt=custom_prompt)

# 질의 루프 
print("🚗 자동차 매뉴얼 기반 RAG QA 시스템 입니다. 알고자 하는 차종을 함께 입력해주세요.")
print("❓ 질문을 입력하세요 ('exit' 입력 시 종료 됩니다.)\n")

while True:
    query = input("질문: ").strip()
    if query.lower() in ["exit", "quit"]:
        break
    
    # 질문에서 차종 추출 
    known_models = ["아반떼", "싼타페", "아이오닉5"]
    matched_model = next((m for m in known_models if m in query), "미지정")
    
    # 유사 문서 검색
    relevant_docs = db.similarity_search(query, k=3)
    
     # 차종종 일치 우선 필터링
    filtered_docs = [doc for doc in relevant_docs if doc.metadata.get("model") == matched_model]
    selected_docs = filtered_docs[:3] if filtered_docs else relevant_docs[:3]
    
    # 차종(model) 정보 추출
    model_count = {}
    for doc in relevant_docs:
        model = doc.metadata.get("model", "미지정")
        model_count[model] = model_count.get(model, 0) + 1
    most_common_model = max(model_count, key=model_count.get)

    # context 구성(출처포함)
    context_blocks = []
    for doc in selected_docs:
        pages = doc.metadata.get("pages", [])
        model = doc.metadata.get("model", "")
        citation = f"[page {pages[0] if pages else '?'} / model: {model}]"
        context_blocks.append(f"{citation}\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    # QA 실행
    result = llm_chain.invoke({
        "query": query,
        "context": context,
        "model": most_common_model
    })

    # 출력
    print("\n💬 답변:")
    print(result["text"])

    print("\n📄 참고 문서:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"[{i}] {doc.metadata.get('type', '')} / pages: {doc.metadata.get('pages', [])} / model: {doc.metadata.get('model', '')}")
        print(doc.page_content[:300])
        print("-" * 40)
