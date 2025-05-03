from modules import load_embedding, load_db, load_llm_chain

embedding = load_embedding()
db = load_db(embedding)
llm_chain = load_llm_chain()
