import streamlit as st
import requests
from PIL import Image
import re  # 줄바꿈 처리를 위한 정규표현식 사용


# UI 설정
st.set_page_config(
    page_title="현대차 설명서 기반 챗봇",
    layout="wide"
)
st.title("🚗 현대자동차 매뉴얼 기반 RAG QA 시스템")

# 카테고리 지정
category_map = {
    "세단": ["아반떼", "소나타", "그랜저"],
    "SUV": ["싼타페", "투싼", "스타리아"],
    "전기차": ["아이오닉9", "아이오닉5"],
    "경차": ["캐스퍼"]
}

category = st.selectbox("🚘 차량 카테고리를 선택 해주세요", list(category_map.keys()))
model_options = category_map[category]
car_model = st.selectbox("🚗 카테고리에 맞는 가능한 차량 모델을 선택 해주세요", model_options)

# 텍스트 또는 이미지 선택
query_mode = st.radio("질문 유형을 선택하세요", ("텍스트 질문", "이미지 업로드"))

# 히스토리 저장
if "history" not in st.session_state:
    st.session_state.history = []

# 텍스트 질문
if query_mode == "텍스트 질문":
    query = st.text_input("❓ 질문을 입력하세요 (예: 싼타페의 시동이 안 걸릴 때 조치 방법은)")

    if query:
        try:
            with st.spinner("💬 답변 생성 중입니다. 잠시만 기다려주세요..."):
                response = requests.post(
                    "http://backend:8000/query",
                    json={"query": query, "model": car_model}
                )
                response.raise_for_status()
                result = response.json()

                answer = result.get("answer", "")
                sources = result.get("sources", [])

                answer_with_linebreaks = re.sub(r'([.?!])\s*', r'\1<br>', answer)

                st.session_state.history.append((query, answer))

                st.markdown("### 💬 답변")
                st.markdown(answer_with_linebreaks, unsafe_allow_html=True)

                st.markdown("### 📄 참고 문서")
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**[{i}] page {src['page']} / model: {src['model']}**")

        except Exception as e:
            st.error(f"FastAPI 요청 실패: {e}")

# 이미지 업로드
else:
    image_file = st.file_uploader("이미지를 업로드 해주세요. 이미지를 분석하여 질문 및 답변을 생성합니다.", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="업로드된 이미지", use_column_width=False, width=300)

        try:
            with st.spinner("🧠 업로드된 이미지를 분석하고 질문을 생성 중입니다..."):
                files = {"image": image_file.getvalue()}
                response = requests.post(
                    "http://backend:8000/image-query",
                    files=files,
                    data={"model": car_model}
                )
                response.raise_for_status()
                result = response.json()

            query = result.get("generated_question", "")
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            st.session_state.history.append((query, answer))

            st.markdown("### 🧠 생성된 질문")
            st.write(query)

            with st.spinner("📘 매뉴얼 문서를 기반으로 답변을 생성 중입니다..."):
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
