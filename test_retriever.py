from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import TextEmbedder
from src.retriever import retrieve_top_k

docs = load_all_documents()
cleaned_docs = clean_documents(docs)

# IMPORTANT: use all chunks now, not only first 50
chunks = chunk_documents(cleaned_docs, strategy="fixed", chunk_size=500, overlap=100)
texts = [chunk["text"] for chunk in chunks]

embedder = TextEmbedder()
chunk_embeddings = embedder.embed_texts(texts)

query = "What is the theme of the 2025 budget statement?"
results = retrieve_top_k(
    query=query,
    embedder=embedder,
    chunk_embeddings=chunk_embeddings,
    chunk_docs=chunks,
    top_k=5,
    use_expansion=True
)

print(f"Total chunks indexed: {len(chunks)}")
print(f"Query: {query}")
print("\nTop retrieval results:\n")

for i, result in enumerate(results, 1):
    print(f"Rank: {i}")
    print(f"Source Type: {result['source_type']}")
    print(f"Source Name: {result['source_name']}")
    print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
    print(f"Keyword Overlap: {result['keyword_overlap']:.4f}")
    print(f"Final Score: {result['final_score']:.4f}")
    print(f"Text Preview: {result['text'][:300]}")
    print("-" * 60)