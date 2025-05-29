from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from .modules import load_embedding, load_db, load_llm_chain

from PIL import Image
import tempfile 


router = APIRouter()
embedding = load_embedding()
llm_chain = load_llm_chain()

class QueryRequest(BaseModel):
    query: str
    model: str

class SourceInfo(BaseModel):
    page: int
    model: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]

# 텍스트단단
@router.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    model_key = {
        "아반떼": "avante",
        "싼타페": "santafe",
        "투싼": "tucson",
        "캐스퍼": "casper",
        "스타리아": "staria",
        "그랜저": "grandeur",
        "소나타": "sonata",
        "아이오닉9": "ionic9",
        "아이오닉5": "ionic5"
    }.get(req.model)

    if not model_key:
        raise HTTPException(status_code=400, detail="지원하지 않는 차량 모델입니다.")

    try:
        db = load_db(model_key, embedding)
    except Exception:
        raise HTTPException(status_code=500, detail="인덱스 로딩 실패")

    docs = db.similarity_search(req.query, k=10)
    filtered = [doc for doc in docs if doc.metadata.get("model") == req.model]
    selected = filtered[:3] if filtered else docs[:3]

    if not selected:
        return QueryResponse(answer="해당 차량에 대한 정보가 없습니다.", sources=[])

    context_blocks = []
    model_count = {}

    for doc in selected:
        model = doc.metadata.get("model", "미지정")
        pages = doc.metadata.get("pages", ["?"])
        model_count[model] = model_count.get(model, 0) + 1
        citation = f"[page {pages[0]} / model: {model}]"
        context_blocks.append(f"{citation}\n{doc.page_content}")

    context = "\n\n".join(context_blocks)
    most_common_model = max(model_count, key=model_count.get, default="미지정")

    result = llm_chain.invoke({
        "query": req.query,
        "context": context,
        "model": most_common_model
    })

    answer_text = result.split("답변:")[-1].strip() if "답변:" in result else result.strip()
    sources = [{"page": doc.metadata.get("pages", ["?"])[0], "model": doc.metadata.get("model", "미지정")} for doc in selected]

    return QueryResponse(answer=answer_text, sources=sources)

