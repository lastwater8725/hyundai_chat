import os
import json
import re
from glob import glob
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LAParams 

input_dir = "./data/car_manual_data"
output_dir = "./data/parsed/pdfminer"

# 글자 감지, 타입 분류, 경고 분류
def parse_pdf(pdf_path):
    laparams = LAParams()
    pages_data = []
    
    list_pattern = r"^\s*(\d+[\.\)]|[\-\•\○])\s+"
    warning_keywords = ["주의", "경고", "WARNING", "CAUTION", "⚠️"]
    
    for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams= laparams), start=1):
        blocks = []
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                text = element.get_text().strip()
                if not text:
                    continue
                
                # 글자크기 계산
                font_size = []
                for text_line in element:
                    for char in text_line:
                        if hasattr(char, 'size'):
                            font_size.append(char.size)
                avg_font_size = sum(font_size) / len(font_size) if font_size else 0
                
                # 기본 타입 (title, subtitle, paragraph)
                if avg_font_size > 15:
                    block_type = "title"
                elif avg_font_size > 12:
                    block_type = "subtitle"
                else:
                    block_type = "paragraph"
            
                # 경고 분류
                if any(keyword in text for keyword in warning_keywords):
                    block_type = "warning"
                    
                # 멀티라인 리스트 분류 example: 1.1.1, 1) 2)
                split_lines = text.split("\n")
                for line in split_lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    current_block_type = block_type
                    
                    if re.match(list_pattern, line):
                        current_block_type = "list"
                        
                    blocks.append({
                        "type": current_block_type,
                        "bbox": [element.x0, element.y0, element.x1, element.y1],  
                        "text": line
                    })
                    
                    
        # 🔥 warning 블록 병합 추가
        merged_blocks = []
        prev_block = None
        for block in blocks:
            if prev_block and block["type"] == "warning" and prev_block["type"] == "warning" and block["bbox"] == prev_block["bbox"]:
                prev_block["text"] += " " + block["text"]
            else:
                merged_blocks.append(block)
                prev_block = block

        pages_data.append({
            "page": page_num,
            "blocks": merged_blocks
        })
        
    return pages_data
                    
def save_json(data, file_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {output_path}")
                                    

def read_pdf(input_dir, output_dir):
    pdf_files = glob(os.path.join(input_dir, "*.pdf"))
    print(f"PDF 파일 개수: {len(pdf_files)}") 
    
    for pdf_file in pdf_files:
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        output_name = f"{base_name}_pdfminer.json"
        parsed = parse_pdf(pdf_file)
        save_json(parsed, output_name, output_dir)

if __name__== "__main__":
    read_pdf(input_dir, output_dir) 