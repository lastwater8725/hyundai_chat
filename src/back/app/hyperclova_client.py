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
    {"role": "user", "content": {
        "type": "text",
        "text": (
           "이 사진은 자동차와 관련되어 있습니다..\n"
        "경고등, 파손과 같은 특이한 점이 보이면 다음과 같은 형식으로 정확히 답변해주세요.\n\n"
        "특이한 점은 보통 빨간색, 노란색등 강조된 색으로 표시됩니다."
        "특이한 점만 모양을 포함해서 답변해주세요"
        "- 특이사항 종류 및 모양: 예) 스티어링 휠 모양 경고등, 엔진 모양 경고등, 브레이크, 타이어 압력 등\n"
        "- 불빛 색상(존재시): 빨간색, 노란색, 없음 등\n"
        "- 해석: 무엇을 의미하는지, 혹은 알 수 없다면 '모르겠다'라고 명확히 작성\n\n"
        "스티어링 휠은 동그란 모양, 엔진은 직사각형 등으로 존재합니다."
        "색상이 여러개 존재할시 특이사항과 관련된 색상으로 작성해주세요"
        "예시: \n"
        "- 특이사항 종류 및 모양: 스티어링 휠 모양\n"
        "- 불빛 색상: 빨간색\n"
        "- 해석: 스티어링 휠 모양 빨간색 경고등이 존재합니다 이에 대한 답변을 문서 기반으로 찾아주세요요.\n\n"
        "해당 형식 외의 문장은 절대 포함하지 말고, 반드시 세 줄만 출력하세요."
        "종류 및 경고등 모양, 색상은 반드시 존재해야합니다."
        )
    }}
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
    print("\n[생성된 답변]", result)
