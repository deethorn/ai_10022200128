#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

from pathlib import Path
import pandas as pd
from pypdf import PdfReader

from src.config import CSV_FILE, PDF_FILE


def load_csv_data(csv_path: Path = CSV_FILE):
    df = pd.read_csv(csv_path)
    documents = []

    for i, row in df.iterrows():
        row_text_parts = []
        for column in df.columns:
            value = row[column]
            row_text_parts.append(f"{column}: {value}")

        row_text = " | ".join(row_text_parts)

        documents.append({
            "doc_id": f"csv_row_{i}",
            "source_type": "csv",
            "source_name": csv_path.name,
            "row_number": int(i),
            "text": row_text
        })

    return documents


def load_pdf_data(pdf_path: Path = PDF_FILE):
    reader = PdfReader(str(pdf_path))
    documents = []

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text(extraction_mode="layout")
        except TypeError:
            page_text = page.extract_text()

        if page_text is None:
            page_text = ""

        page_text = page_text.strip()

        if page_text:
            documents.append({
                "doc_id": f"pdf_page_{i+1}",
                "source_type": "pdf",
                "source_name": pdf_path.name,
                "page_number": i + 1,
                "text": page_text
            })

    return documents


def load_all_documents():
    csv_docs = load_csv_data()
    pdf_docs = load_pdf_data()
    return csv_docs + pdf_docs
