#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import TextEmbedder
from src.llm_generator import LLMGenerator
from src.pipeline import run_rag_pipeline

docs = load_all_documents()
cleaned_docs = clean_documents(docs)
chunks = chunk_documents(cleaned_docs, strategy="fixed", chunk_size=500, overlap=100)

texts = [chunk["text"] for chunk in chunks]

embedder = TextEmbedder()
chunk_embeddings = embedder.embed_texts(texts)

llm = LLMGenerator(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")

query = "What is the theme of the 2025 budget statement?"

result = run_rag_pipeline(
    query=query,
    embedder=embedder,
    chunk_embeddings=chunk_embeddings,
    chunk_docs=chunks,
    llm=llm,
    top_k=5
)

print("QUERY:")
print(result["query"])

print("\nANSWER SOURCE:")
print(result["answer_source"])

print("\nFINAL ANSWER:")
print(result["final_answer"])

print("\nTOP RETRIEVED CHUNK:")
print(result["retrieved_chunks"][0]["text"][:500])
