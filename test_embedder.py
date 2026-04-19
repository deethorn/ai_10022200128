from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import TextEmbedder

docs = load_all_documents()
cleaned_docs = clean_documents(docs)
chunks = chunk_documents(cleaned_docs, strategy="fixed", chunk_size=500, overlap=100)

sample_chunks = chunks[:10]
sample_texts = [chunk["text"] for chunk in sample_chunks]

embedder = TextEmbedder()

chunk_embeddings = embedder.embed_texts(sample_texts)
query_embedding = embedder.embed_query("What is the theme of the 2025 budget statement?")

print("Number of sample chunks:", len(sample_texts))
print("Chunk embeddings shape:", chunk_embeddings.shape)
print("Query embedding shape:", query_embedding.shape)
print("First chunk preview:", sample_texts[0][:200])