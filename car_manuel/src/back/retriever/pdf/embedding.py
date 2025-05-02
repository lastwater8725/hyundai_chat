import os 
import json
from glob import glob
from tqdm import tqdm
import torch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  


input_dir = "./data/parsed/pdfminer/chunker"
output_dir = "./data/parsed/pdfminer/embedding/faiss_index"
os.makedirs(output_dir, exist_ok=True)

# 임베딩 모델 로드
embedding_model_name = "BAAI/bge-m3"
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)


def load_chunks(chunk_path):
    docs = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = data["text"]
            metadata = {
                "type": data.get("type", ""),
                "pages": data.get("source_pages", []),
                "model": data.get("model", "미지정")
            }
            docs.append(Document(page_content=content, metadata=metadata))
    
    if docs:
        print(f"예시 문서 metadata: {docs[0].metadata}")
    
    return docs


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def main():
    chunk_files = glob(os.path.join(input_dir, "*_chunks.json"))
    print(f"청크 파일 개수: {len(chunk_files)}")
    
    all_docs = []
    for chunk_file in chunk_files:
        docs = load_chunks(chunk_file)
        all_docs.extend(docs)
        
    print(f"전체 문서 개수: {len(all_docs)}")
    
    # FAISS 인덱스 생성 배치로로
    batch_size = 8
    index = None
    
    for batch in tqdm(list(chunked(all_docs, batch_size)), desc="embedding"):
        if not batch:
            continue
        if index is None:
            index = FAISS.from_documents(batch, embedding_model)
        else:
            index.add_documents(batch)
        
        # 캐시 정리
        torch.cuda.empty_cache()
        
    index.save_local(output_dir)
    print(f" 인덱스 저장 완료: {output_dir}")

if __name__ == "__main__":
    main()
