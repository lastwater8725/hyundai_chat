import streamlit as st
from pathlib import Path
import sys

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
st.header('현재 등록되어 있는 모델은 아반떼, 싼타페, 투싼, 캐스퍼, 스타리아, 그랜저, 소나타 입니다.')

query = st.text_input("❓ 질문을 입력하세요 (예: 싼타페의 뒷유리 와이퍼가 작동하지 않을 때 확인할 사항은) exit를 누르면 종료됩니다.")

known_models = ["아반떼", "싼타페", "투싼", "캐스퍼", "스타리아", "그랜저", "소나타"]
matched_model = next((m for m in known_models if m in query), "미지정")

if query:
    relevant_docs = db.similarity_search(query, k=10)
    filtered_docs = [doc for doc in relevant_docs if doc.metadata.get("model") == matched_model]
    selected_docs = filtered_docs[:3] if filtered_docs else relevant_docs[:3]

    context_blocks = []
    model_count = {}
    for i, doc in enumerate(selected_docs):
        model = doc.metadata.get("model", "미지정")
        pages = doc.metadata.get("pages", ["?"])
        model_count[model] = model_count.get(model, 0) + 1
        citation = f"[page {pages[0]} / model: {model}]"
        context_blocks.append(f"{citation}\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    most_common_model = max(model_count, key=model_count.get)
    result = llm_chain.invoke({
        "query": query,
        "context": context,
        "model": most_common_model
    })

    full_output = result["text"]
    if "답변:" in full_output:
        answer = full_output.split("답변:")[-1].strip()
    else:
        answer = full_output.strip()

    st.markdown("### 💬 답변")
    st.write(answer)

    st.markdown("### 📄 참고 문서")
    for i, doc in enumerate(selected_docs, 1):
        model = doc.metadata.get("model", "")
        pages = doc.metadata.get("pages", ["?"])
        st.markdown(f"**[{i}] page {pages[0]}, model: {model}**")
        st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
