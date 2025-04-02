import os 
from src.doclayout_paser import parse_images
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

###전처리 실행 코드드

# 모델 다운로드
model_path = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
model = YOLOv10(model_path)

# 이미지 변환
def pdf_to_images(pdf_path, output_dir="data/images"):
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=300)
    
    image_paths = []
    for i, page in enumerate(pages):
        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        page.save(image_path, "JPEG")
        image_paths.append(image_path)
    return image_paths

#실행
if __name__ == "__main__":
    pdf_path = "./data/car_manual/avante_Owner's_Manual.pdf"
    
    print("pdf to img 변환중")
    image_paths = pdf_to_images(pdf_path)
    
    print("doclayout-yolo 및 ocr 파싱 중 ")
    parse_images(image_paths, visualize=True)
    
    print("파싱 완료. 결과 저장 경로 -> data/parsed/doclayout_yolo_result.json")
    print("시각화 저장: data/visualized/")