import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 경로
embedding_path = "data/embeddings/chunk_embeddings.pt"
index_save_path = "data/embeddings/faiss_index"
embedding_model_name = "BAAI/bge-m3"

# 1. 임베딩 로드
def load_embeddings(path):
    data = torch.load(path)
    embeddings = data["embeddings"].cpu().numpy()
    metadata = data["metadata"]
    return embeddings, metadata

# 2. Document 객체로 변환
def to_documents(metadata):
    documents = []
    for item in metadata:
        text = item["text"]
        meta = {
            "type": item.get("type"),
            "bbox": item.get("bbox"),
            "source_image": item.get("source_image")
        }
        documents.append(Document(page_content=text, metadata=meta))
    return documents

if __name__ == "__main__":
    print("🔹 임베딩 및 메타데이터 불러오는 중...")
    embeddings, metadata = load_embeddings(embedding_path)
    documents = to_documents(metadata)

    print("🔹 임베딩 모델 로딩 중...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    print("🔹 LangChain용 FAISS 인덱스 생성 중...")
    vectorstore = FAISS.from_documents(documents, embedding_model)

    print(f"💾 인덱스 저장 중 → {index_save_path}")
    vectorstore.save_local(index_save_path)
    print("✅ 완료")
