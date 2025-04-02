import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 경로 설정
index_path = "data/embeddings/faiss_index"
embed_model_name = "BAAI/bge-m3"
llm_model_name = "nlpai-lab/kullm3"

# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# 벡터 db 불러오기
db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

# retriver
retriever = db.as_retriever(search_kwargs={"k": 3})

# 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name).to("cuda")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    max_new_tokens=512,
)

llm = HuggingFacePipeline(pipeline=pipe)

#프롬프트 템플릿(명시)
template = """너는 현대, 기아 자동차 메뉴얼 기반 ai야 문서내용을 참고하여
질문에 답변해줘, 또한 어떤 차종인지 얘기해줘
질문: {question}
답변:
"""
prompt = PromptTemplate.from_template(template)

#qa_chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 실행
if __name__ == "__main__":
    while True:
        query = input("❓ 질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            break
        answer = qa_chain.run(query)
        print(f"\n💬 답변: {answer}\n")