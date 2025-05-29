import streamlit as st
import requests
from PIL import Image

# UI 설정
st.set_page_config(page_title="현대차 매뉴얼 QA", layout="wide")
st.title("🚗 현대자동차 매뉴얼 기반 RAG QA 시스템")

# 카테고리 지정
category_map = {
    "세단" : ["아반떼", "소나타", "그랜저"],
    "SUV" : ["싼타페", "투싼", "스타리아"],
    "전기차" : ["아이오닉9", "아이오닉5"],
    "경차": ["캐스퍼"]
}

category = st.selectbox("🚘 차량 카테고리를 선택 해주세요", list(category_map.keys()))
model_options = category_map[category]
car_model = st.selectbox("🚗 가능한 차량 모델 중 선택 해주세요", model_options) 

# 텍스트 또는 이미지 선택
query_mode = st.radio("질문 유형을 선택하세요", ("텍스트 질문", "이미지 업로드"))

# 히스토리 저장
if "history" not in st.session_state:
    st.session_state.history = []

if query_mode == "텍스트 질문":
    query = st.text_input("❓ 질문을 입력하세요 (예: 싼타페의 시동이 안 걸릴 때 조치 방법은)")

    if query:
        # FastAPI로 요청 전송
        try:
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": query, "model": car_model}
            )
            response.raise_for_status()
            result = response.json()

            # 답변 및 출처 출력
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.session_state.history.append((query, answer))

            st.markdown("### 💬 답변")
            st.write(answer)

            st.markdown("### 📄 참고 문서")
            for i, src in enumerate(sources, 1):
                st.markdown(f"**[{i}] page {src['page']} / model: {src['model']}**")

        except Exception as e:
            st.error(f"FastAPI 요청 실패: {e}")

else:
    image_file = st.file_uploader("이미지를 업로드 해주세요", type=["jpg", "jpeg", "png"])
    
    if image_file:
        image = image.open(image_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        
        try:
            files = {"image": image_file.getvalue()}
            response = requests.post(
                "http://localhost:8000/image-query",
                files=files,
                data={"model": car_model}
            )
            response.raise_for_status()
            result = response.json()

            query = result.get("generated_query", "")
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.session_state.history.append((query, answer))

            st.markdown("### 🧠 생성된 질문")
            st.write(query)

            st.markdown("### 💬 답변")
            st.write(answer)

            st.markdown("### 📄 참고 문서")
            for i, src in enumerate(sources, 1):
                st.markdown(f"**[{i}] page {src['page']} / model: {src['model']}**")

        except Exception as e:
            st.error(f"이미지 질의 처리 실패: {e}")


# 최근 질문 표시
if st.session_state.history:
    st.markdown("### 📚 이전 질문")
    for q, a in reversed(st.session_state.history[-5:]):
        st.markdown(f"**❓ {q}**")
        st.markdown(f"**💬 {a}**")
        st.markdown("---")
