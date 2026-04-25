#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import TextEmbedder
from src.vector_store import VectorStore

docs = load_all_documents()
cleaned_docs = clean_documents(docs)
chunks = chunk_documents(cleaned_docs, strategy="fixed", chunk_size=500, overlap=100)

sample_chunks = chunks[:50]
sample_texts = [chunk["text"] for chunk in sample_chunks]

embedder = TextEmbedder()
chunk_embeddings = embedder.embed_texts(sample_texts)

dimension = chunk_embeddings.shape[1]
vector_store = VectorStore(dimension=dimension)
vector_store.add_embeddings(chunk_embeddings, sample_chunks)

query = "What is the theme of the 2025 budget statement?"
query_embedding = embedder.embed_query(query)

results = vector_store.search(query_embedding, top_k=5)

print("Embedding dimension:", dimension)
print("Number of indexed chunks:", len(sample_chunks))
print("Top results:\n")

for result in results:
    print(f"Rank: {result['rank']}")
    print(f"Distance: {result['faiss_distance']}")
    print(f"Source: {result['source_name']}")
    print(f"Text preview: {result['text'][:200]}")
    print("-" * 50)
