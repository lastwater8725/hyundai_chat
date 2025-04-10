import os
from pathlib import Path
from PIL import Image
import numpy as np
import json
import easyocr

from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

#모델 불러오기
# Hugging Face에서 모델 직접 다운로드하여 경로 지정
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(model_path)
reader = easyocr.Reader(['ko'], gpu=False)

def parse_image_with_doclayout_yolo(image_path):
    image = Image.open(image_path).convert("RGB")
    
    det_res = model.predict(
        image_path,
        imgsz=1024,
        conf=0.2,
        device="cuda:0"
    )
    
    blocks = []
    boxes = det_res[0].boxes.xyxy.cpu().numpy()
    classes = det_res[0].boxes.cls.cpu().numpy()
    names = det_res[0].names
    confidences = det_res[0].boxes.conf.cpu().numpy()
    
    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls_id)]
        cropped = image.crop((x1,y1,x2,y2))
        
        cropped_np = np.array(cropped)
        result = reader.readtext(cropped_np)
        texts = [line[1] for line in result] if result else []
        
        blocks.append({
            "type": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "text": texts
        })
        
    return blocks

# 여러 이미지 처리 및 저장
def parse_images(image_paths, output_path="data/parsed/doclayout_yolo_result.json"):
    all_results = []
    for path in image_paths:
        page_data = {
            "image": path,
            "blocks": parse_image_with_doclayout_yolo(path)
        }
        all_results.append(page_data)

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_path}")