import streamlit as st
import requests

# UI 설정
st.set_page_config(page_title="현대차 매뉴얼 QA", layout="wide")
st.title("🚗 현대자동차 매뉴얼 기반 RAG QA 시스템")

known_models = ["아반떼", "싼타페", "투싼", "캐스퍼", "스타리아", "그랜저", "소나타"]
st.warning(f"현재 지원하는 차량 모델은 {', '.join(known_models)}입니다. 질문에 차량명을 포함해주세요.")

# 히스토리 저장
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("❓ 질문을 입력하세요 (예: 싼타페의 시동이 안 걸릴 때 조치 방법은)")

if query:
    # 차량 모델 자동 추출
    matched_model = next((m for m in known_models if m in query), None)

    if not matched_model:
        st.error("❌ 차량 모델을 질문에 포함시켜 주세요.")
    else:
        # FastAPI로 요청 전송
        try:
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": query, "model": matched_model}
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

# 최근 질문 표시
if st.session_state.history:
    st.markdown("### 📚 이전 질문")
    for q, a in reversed(st.session_state.history[-5:]):
        st.markdown(f"**❓ {q}**")
        st.markdown(f"**💬 {a}**")
        st.markdown("---")
