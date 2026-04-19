from src.data_loader import load_csv_data, load_pdf_data, load_all_documents

csv_docs = load_csv_data()
pdf_docs = load_pdf_data()
all_docs = load_all_documents()

print("CSV documents:", len(csv_docs))
print("PDF documents:", len(pdf_docs))
print("All documents:", len(all_docs))

if csv_docs:
    print("\nFirst CSV document:")
    print(csv_docs[0])

if pdf_docs:
    print("\nFirst PDF document:")
    print(pdf_docs[0]["text"][:500])