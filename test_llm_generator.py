from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import TextEmbedder
from src.retriever import retrieve_top_k
from src.prompt_builder import build_rag_prompt
from src.llm_generator import LLMGenerator

docs = load_all_documents()
cleaned_docs = clean_documents(docs)
chunks = chunk_documents(cleaned_docs, strategy="fixed", chunk_size=500, overlap=100)

texts = [chunk["text"] for chunk in chunks]

embedder = TextEmbedder()
chunk_embeddings = embedder.embed_texts(texts)

query = "What is the theme of the 2025 budget statement?"

retrieved_chunks = retrieve_top_k(
    query=query,
    embedder=embedder,
    chunk_embeddings=chunk_embeddings,
    chunk_docs=chunks,
    top_k=5,
    use_expansion=True
)

prompt = build_rag_prompt(query, retrieved_chunks, max_chunks=2, max_characters=1200)

llm = LLMGenerator(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
answer = llm.generate_answer(prompt)

print("QUESTION:")
print(query)
print("\nANSWER:")
print(answer)