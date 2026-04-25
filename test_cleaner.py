#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

from src.data_loader import load_all_documents
from src.cleaner import clean_documents

docs = load_all_documents()
cleaned_docs = clean_documents(docs)

print("Original documents:", len(docs))
print("Cleaned documents:", len(cleaned_docs))

print("\nFirst original document:")
print(docs[0]["text"])

print("\nFirst cleaned document:")
print(cleaned_docs[0]["text"])

print("\nFirst cleaned PDF preview:")
for doc in cleaned_docs:
    if doc["source_type"] == "pdf":
        print(doc["text"][:500])
        break
