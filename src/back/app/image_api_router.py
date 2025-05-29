from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List
from .modules import load_embedding, load_db, load_llm_chain
from .hyperclova_client import generate_question_from_image

router = APIRouter()
embedding = load_embedding()
llm_chain = load_llm_chain()

class SourceInfo(BaseModel):
    page: int
    model: str
    
class ImageQueryResponse(BaseModel):
    generated_question: str
    answer: str
    sources: List[SourceInfo]
    
@router.post("/image-query", response_model=ImageQueryResponse)
async def image_query(
    image: UploadFile = File(...),
    model: str = Form(...)
):
    model_key_map = {
        "아반떼": "avante",
        "싼타페": "santafe",
        "투싼": "tucson",
        "캐스퍼": "casper",
        "스타리아": "staria",
        "그랜저": "grandeur",
        "소나타": "sonata",
        "아이오닉9": "ionic9",
        "아이오닉5": "ionic5"
    }
    
    model_key = model_key_map.get(model)
    if not model_key:
        raise HTTPException(status_code=400, detail="지원하지 않는 차량 모델입니다.")
    
    # 이미지에서 질문 생성
    try:
        image_bytes = await image.read()
        generated_query = generate_question_from_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 실패(질문 생성 실패): {e}")
    
    # 유사도 검색 및 답변 생성
    try:
        db = load_db(model_key, embedding)
        docs = db.similarity_search(generated_query, k=10)
    except Exception as e:
        raise HTTPException(status_code=500, detail="인덱스 로딩 또는 검색 실패")
    
    filtered_docs = [doc for doc in docs if doc.metadata.get("model") == model]
    selected_docs = filtered_docs[:3] if filtered_docs else docs[:3]
    
    if not selected_docs:
        return ImageQueryResponse(
            generated_question=generated_query,
            answer="해당 차량에 대한 정보가 없습니다.",
            sources=[]
        )
        
    context_blocks = []
    model_count = {}
    
    for doc in selected_docs:
        doc_model = doc.metadata.get("model", "미지정")
        pages = doc.metadata.get("pages", ["?"])
        model_count[doc_model] = model_count.get(doc_model, 0) + 1
        citation = f"[page {pages[0]} / model: {doc_model}]"
        context_blocks.append(f"{citation}\n{doc.page_content}")

    context = "\n\n".join(context_blocks)
    most_common_model = max(model_count, key=model_count.get)

    result = llm_chain.invoke({
        "query": generated_query,
        "context": context,
        "model": most_common_model
    })

    answer_text = result.split("답변:")[-1].strip() if "답변:" in result else result.strip()
    sources = [{"page": doc.metadata.get("pages", ["?"])[0], "model": doc.metadata.get("model", "미지정")} for doc in selected_docs]

    return ImageQueryResponse(generated_query=generated_query, answer=answer_text, sources=sources)