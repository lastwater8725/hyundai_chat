import streamlit as st
from pathlib import Path
import sys
import os

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODULE_PATH = BASE_DIR / "back" / "retriever" / "pdf"
sys.path.append(str(MODULE_PATH))

# --- modules.py에서 필요한 함수 불러오기 ---
from modules import load_embedding, load_db, load_llm_chain

# --- 초기화 ---
embedding = load_embedding()
db = load_db(embedding)
llm_chain = load_llm_chain()

# --- Streamlit UI ---
st.set_page_config(page_title="현대차 매뉴얼 QA", layout="wide")
st.title("🚗 현대자동차 매뉴얼 기반 RAG QA 시스템")

# 등록모델 모델 안내
known_models = ["아반떼", "싼타페", "투싼", "캐스퍼", "스타리아", "그랜저", "소나타"]
st.warning(f"현재 지원하는 차량 모델은 {', '.join(known_models)}입니다. 질문시 차량을 함께 질문해주세요")

# 세션 상태에 질문 히스토리 저장
if "history" not in st.session_state:
    st.session_state.history = []    
    
query = st.text_input("❓ 질문을 입력하세요 (예: 싼타페의 뒷유리 와이퍼가 작동하지 않을 때 확인할 사항은)")


if query:
    matched_model = next((m for m in known_models if m in query), "미지정")
    # 유사 문서 검색
    relevant_docs = db.similarity_search(query, k=10)
    
    # 모델 일치 우선 필터링
    filtered_docs = [doc for doc in relevant_docs if doc.metadata.get("model") == matched_model]
    selected_docs = filtered_docs[:3] if filtered_docs else relevant_docs[:3]

    # 참고 문서 없을 경우 메시지 출력
    if not selected_docs:
        st.error("❌ 관련 차량 문서가 없습니다. 질문 내용을 다시 확인해주세요.")
    else:    
        # 생성성
        context_blocks = []
        model_count = {}
        for i, doc in enumerate(selected_docs):
            model = doc.metadata.get("model", "미지정")
            pages = doc.metadata.get("pages", ["?"])
            model_count[model] = model_count.get(model, 0) + 1
            citation = f"[page {pages[0]} / model: {model}]"
            context_blocks.append(f"{citation}\n{doc.page_content}")
        context = "\n\n".join(context_blocks)

        # 가장 많은 모델 기준
        if model_count:
            most_common_model = max(model_count, key=model_count.get)
        else:
            most_common_model = "미지정"

        
        # qa
        result = llm_chain.invoke({
            "query": query,
            "context": context,
            "model": most_common_model
        })

        # 답변만 추출
        full_output = result["text"]
        answer = full_output.split("답변:")[-1].strip() if "답변:" in full_output else full_output.strip()

        # 히스토리에 저장
        st.session_state.history.append((query, answer))

        # 답변 출력
        st.markdown("### 💬 답변")
        st.write(answer)

        st.markdown("### 📄 참고 문서")
        for i, doc in enumerate(selected_docs, 1):
            model = doc.metadata.get("model", "")
            pages = doc.metadata.get("pages", ["?"])
            st.markdown(f"**[{i}] page {pages[0]}, model: {model}**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

#  이전 질문 히스토리
if st.session_state.history:
    st.markdown("### 📚 이전 질문")
    for q, a in reversed(st.session_state.history[-5:]):  # 최근 5개만
        st.markdown(f"**❓ {q}**")
        st.markdown(f"**💬 {a}**")
        st.markdown("---")