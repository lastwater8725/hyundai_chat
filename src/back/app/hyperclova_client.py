from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

def generate_question_from_image(image_path: str, prompt: str = "이미지를 보고 질문 하나 생성해줘.") -> str:
    model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

    # 모델 및 프로세서 로드
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")

    # 이미지 로드
    image = Image.open(image_path).convert("RGB")

    # 입력 변환
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    # 질문 생성
    generate_ids = model.generate(**inputs, max_new_tokens=64)
    question = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    return question

if __name__ == "__main__":
    img_path = "sample_image.jpg"  # 테스트용 이미지 파일 경로
    result = generate_question_from_image(img_path)
    print("\n[생성된 질문]", result)
