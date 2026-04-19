from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents

docs = load_all_documents()
cleaned_docs = clean_documents(docs)

fixed_chunks = chunk_documents(cleaned_docs, strategy="fixed", chunk_size=500, overlap=100)
paragraph_chunks = chunk_documents(cleaned_docs, strategy="paragraph", chunk_size=500, overlap=100)

print("Cleaned documents:", len(cleaned_docs))
print("Fixed-size chunks:", len(fixed_chunks))
print("Paragraph chunks:", len(paragraph_chunks))

print("\nFirst fixed chunk:")
print(fixed_chunks[0]["text"])

print("\nFirst paragraph chunk:")
print(paragraph_chunks[0]["text"])

print("\nSample PDF fixed chunk:")
for chunk in fixed_chunks:
    if chunk["source_type"] == "pdf":
        print(chunk["text"][:700])
        break

print("\nSample PDF paragraph chunk:")
for chunk in paragraph_chunks:
    if chunk["source_type"] == "pdf":
        print(chunk["text"][:700])
        break