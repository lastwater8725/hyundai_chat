import os 
import json
from glob import glob

input_dir = "./data/parsed/pdfminer"
output_dir = "./data/parsed/pdfminer/chunker"

os.makedirs(output_dir, exist_ok=True)

def load_parsed_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def save_chunks(chunks, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"청크 저장 완료: {file_path}")

def create_chunks(pages_data):
    chunks = []
    current_chunk = {
        "text": "",
        "type": "",
        "source_pages": []
    }
    
    for page in pages_data:
        page_num = page["page"]
        for block in page["blocks"]:
            block_type = block["type"]
            block_text = block["text"].strip()
            
            if not block_text:
                continue
            
            # 제목 등장 시 새로운 청크 시작
            if block_type in ["title", "subtitle"]:
                if current_chunk["text"]:
                    chunks.append(current_chunk)
                current_chunk = {
                    "text": block_text,
                    "type": block_type,
                    "source_pages": [page_num]
                }
            else:
                # 일반 블록은 이어 붙이기
                if current_chunk["text"]:
                    current_chunk["text"] += "\n" + block_text
                else:
                    current_chunk["text"] = block_text
                    
                if page_num not in current_chunk.get("source_pages", []):
                    current_chunk.setdefault("source_pages", []).append(page_num)
                    
    # 마지막 청크 저장
    if current_chunk["text"]:
        chunks.append(current_chunk)
                    
    return chunks
            
def main():
    json_files = glob(os.path.join(input_dir, "*.json"))
    print(f"JSON 파일 개수: {len(json_files)}")
    
    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_name = f"{base_name}_chunks.json"
        
        parsed_data = load_parsed_data(json_file)
        chunks = create_chunks(parsed_data)
        save_chunks(chunks, os.path.join(output_dir, output_name))
        
if __name__ == "__main__":
    main()