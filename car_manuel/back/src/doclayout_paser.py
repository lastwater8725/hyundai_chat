import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import gc

from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from paddleocr import PaddleOCR

#모델 불러오기
# Hugging Face에서 모델 직접 다운로드하여 경로 지정
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(model_path)
ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=False, enable_mkldnn=True)

def parse_image_with_doclayout_yolo(image_path, visualize=False, save_dir="data/visualized"):
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
    
    if visualize:
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            
    for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls_id)]
        cropped = image.crop((x1,y1,x2,y2))
        
        cropped_np = np.array(cropped)
        ocr_result = ocr.ocr(cropped_np, cls=True)
        texts = []
        if ocr_result and isinstance(ocr_result[0], list):
            texts = [{"text": line[1][0], "ocr_confidence": float(line[1][1])} for line in ocr_result[0]]

        
        blocks.append({
            "type": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "text": texts
        })
        
        if visualize:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), f"{label}", fill="blue", font=font)
            if texts:
                draw.text((x1, y2 + 5), texts[0]['text'][:30], fill="green", font=font)
        
    #시각화 이미지 저장
    if visualize:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        image.save(save_path)
        print(f"시각화 저장 -> {save_path}")
        
    return blocks

# 여러 이미지 처리 및 저장
# def parse_images(image_paths, output_path="data/parsed/doclayout_yolo_result.json", visualize=False):
#     all_results = []
#     for path in image_paths:
#         page_data = {
#             "image": path,
#             "blocks": parse_image_with_doclayout_yolo(path, visualize=visualize)
#         }
#         all_results.append(page_data)

#     Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(all_results, f, ensure_ascii=False, indent=2)

#     print(f"저장 완료: {output_path}")

def parse_images(image_paths, output_dir="data/parsed", visualize=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, path in enumerate(image_paths):
        result = {
           "image": path,
            "blocks": parse_image_with_doclayout_yolo(path, visualize=visualize)
        }

        # 파일명 예: page_0001.json
        filename = f"page_{i+1:04}.json"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"저장 완료: {output_path}")
        
        del result
        gc.collect()