from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
import torch


def generate_question_from_image(image_path: str) -> str:
    model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

    # 모델과 프로세서, 토크나이저 로드
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda")

    # Step 1. VLM chat format 구성 (이미지 + 텍스트 프롬프트)
    vlm_chat = [
        {"role": "user", "content": {"type": "image", "image": image_path}},
        {"role": "user", "content": {"type": "text", "text": "사진을 보고 특이사항에 대해서 설명해줘. 경고등과 같은 문제가 있거나 특이한 사항은 자세하게 어떤 모습인지 알려줘줘 자동차와 관련된 사진이야야"}}
    ]

    # Step 2. 이미지/비디오 전처리 (내부적으로 PIL.Image로 자동 변환됨)
    new_chat, all_images, is_video_list = processor.load_images_videos(vlm_chat)
    image_features = processor(all_images, is_video_list=is_video_list)

    # Step 3. 입력 텍스트 구성
    input_ids = tokenizer.apply_chat_template(
        new_chat,
        return_tensors="pt",
        tokenize=True,
        add_generation_prompt=True
    ).to("cuda")

    # Step 4. 모델 추론
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.6,
        temperature=0.5,
        repetition_penalty=1.0,
        **image_features
    )

    # 결과 디코딩
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return result


if __name__ == "__main__":
    image_path = "sample_image.jpg"
    result = generate_question_from_image(image_path)
    print("\n[생성된 질문]", result)
