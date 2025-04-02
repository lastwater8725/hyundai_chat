import os
import json
import torch
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

embedding_path = "data/embeddings/chunk_embeddings.pt"
faiss_index_path = "data/index/faiss.index"
meta_path = "data/index/metadata.json"

def load_embeddings(path):
    data = torch.load(path)
    embeddings = data["embeddings"].cpu().numpy()
    metadata = data["metadata"]
    return embeddings, metadata 

def build_faiss_index(embeddings):
    dim = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dim)      #l2기반 거리
    index.add(embeddings)
    return index 

def save_faiss_index(index, metadata, index_path, meta_path):
    Path(os.path.dirname(index_path)).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    print("임베딩 불러오기")
    embeddings, metadata = load_embeddings(embedding_path)
    
    print("faiss 인덱스 생성 중")
    index = build_faiss_index(embeddings)
    
    print(f"인덱스, 메타데이터 저장 -> {faiss_index_path}")
    save_faiss_index(index, metadata, faiss_index_path, meta_path)
    print("완료요")    