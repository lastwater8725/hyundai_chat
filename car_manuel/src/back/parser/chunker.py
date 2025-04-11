import os 
import json
from pathlib import Path
from tqdm import tqdm

#입력 json
input_path = "data/parsed/doclayout_yolo_result_easyocr.json"
output_path = "data/chunks/chunks.jsonl"

def load_json(input_path):
    with open(input_path, 'r', encoding = "utf-8") as f:
        return json.load(f)
            
def extract_chunks_from_page(page_data):
    chunks = []
    for block in page_data.get("blocks", []):
        texts = block.get("text", [])
        text = " ".join(texts).strip()
        if text:
            chunks.append({
                "text": text,
                "type": block.get("type"),
                "bbox": block.get("bbox"),
                "source_image": page_data.get("image")
            })
    return chunks

def save_chunks(chunks, ouput_path):
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding = "utf-8") as f:
        for chunk in chunks:
            json_line = json.dumps(chunk, ensure_ascii=False)
            f.write(json_line + "\n")

def main():           
    all_chunks = []
    print("파싱된 JSON 파일로부터 chunk 추출 중...")
    
    pages = load_json(input_path)
    for page_data in tqdm(pages):
        chunks = extract_chunks_from_page(page_data)
        all_chunks.extend(chunks)

    print(f"총 {len(all_chunks)}개 chunk 저장 중 → {output_path}")
    save_chunks(all_chunks, output_path)
    print("완료")

if __name__ == "__main__":
    main()