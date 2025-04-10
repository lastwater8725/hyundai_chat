import os
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer 
import torch    

#경로
chunk_path = "data/chunks/chunks.jsonl"
output_path = "data/embeddings/chunk_embeddings.pt"

# 모델 불러오기 및 gpu
model = SentenceTransformer("BAAI/bge-m3")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 불러오기
def load_chunk(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    
#임베딩 저장
def save_embeddings(embeddings, metadata, path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "metadata": metadata}, path)
    
#실행 
def main():
    print("청크 부르기")
    chunks = load_chunk(chunk_path)
    texts = [c["text"] for c in chunks]
    
    print("임베딩 중")
    embeddings = model.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=True)
    
    print(f"임베딩 저장 중 -> {output_path}")
    save_embeddings(embeddings, chunks, output_path)
    print("완료")
    
if __name__ == "__main__":
    main()
