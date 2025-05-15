# CLI에서 실행시 엔트리 포인트

from modules import load_embedding, load_db, load_llm_chain

embedding = load_embedding()
db = load_db(embedding)
llm_chain = load_llm_chain()
