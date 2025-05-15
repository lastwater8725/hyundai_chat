# modules.py — 핵심 기능만 분리
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def load_embedding():
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",       # ko 모델 
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding

def load_db(embedding):
    index_path = './data/parsed/pdfminer/embedding/faiss_index'
    db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    # 서버 배포시에 역직렬화 사용시 위험하므로 서버 배포시엔 유의 수정 필요
    return db

def load_llm_chain():
    llm_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
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

    # === 프롬프트 템플릿 정의 ===
    template = """
    너는 현대자동차 매뉴얼 기반 AI야.
    아래 문서를 참고해서 사용자의 질문에 친절하고 정확하게 답변해줘.
    각 문서는 특정 차량 모델({model})에 대한 내용이야.
    현재 문서에는 아반떼, 싼타페, 투싼, 스타리아, 케스퍼, 그랜저, 소나타에 대한 정보가 포함되어있어
    관련 내용 우선적으로 참고해서 바탕으로 안내해줘.
    참고한 rag문서의 페이지와 차량 이름을 항상 같이 출력해줘.
    rag문서 내용을 명확하게 파악하고 답변해줘.
    문서 내용과 함께 [출처 정보]도 포함되어 있어. 예시처럼 페이지 번호와 차량 모델을 인용해줘:
    예) [page 번호, model: 차종]
    질문한 차량 모델과 문서의 모델명이 일치할 경우 해당 내용을 가장 신뢰도 높은 정보로 간주해서 답변을 구성해줘.
    다른 차량 문서에 유사한 정보가 있을 수 있으니 필요시 참고하고 어떤 문서도 함께 참고했는지 명시해줘
    다른 차량 문서를 참고한 경우, 어떤 문서를 참고했는지도 명확히 문장으로 작성해줘. 그러나 우선적으로 질문과 같은 차량의 문서를 우선적으로 참고해서 답변해
    주어진 문서 내용을 기반으로 실제 조건이나 상황이 명시되지 않은 경우, 답변을 유보하거나 추론된 설명은 "추정"임을 분명히 해줘.
    참고 할 수 있는 문서가 없었다면 "해당 차량에 대한 정보가 없습니다."라고 필수적으로 답변해줘.

    문서 내용:
    {context}

    질문:
    {question}

    답변을 문장으로 완성해줘.
    답변:
"""

    prompt = PromptTemplate(
        input_variables=["context", "query", "model"],
        template=template.replace("{question}", "{query}")
    )

    chain = prompt | llm 
    return chain
