import re
from typing import List, Dict


def fixed_size_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []

    if not text:
        return chunks

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> List[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            if len(sentence) > chunk_size:
                chunks.extend(fixed_size_chunk_text(sentence, chunk_size, overlap))
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def paragraph_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []

    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return fixed_size_chunk_text(text, chunk_size, overlap)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph_parts = split_long_paragraph(paragraph, chunk_size, overlap)

        for part in paragraph_parts:
            candidate = f"{current_chunk}\n\n{part}".strip() if current_chunk else part

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return add_overlap_to_chunks(chunks, overlap)


def add_overlap_to_chunks(chunks: List[str], overlap: int = 200) -> List[str]:
    if not chunks:
        return []

    overlapped_chunks = [chunks[0]]

    for i in range(1, len(chunks)):
        previous_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        overlap_text = previous_chunk[-overlap:] if len(previous_chunk) > overlap else previous_chunk
        merged_chunk = f"{overlap_text}\n{current_chunk}".strip()
        overlapped_chunks.append(merged_chunk)

    return overlapped_chunks


def chunk_document(doc: Dict, strategy: str = "fixed", chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    text = doc.get("text", "")

    if strategy == "fixed":
        text_chunks = fixed_size_chunk_text(text, chunk_size, overlap)
    elif strategy == "paragraph":
        text_chunks = paragraph_chunk_text(text, chunk_size, overlap)
    else:
        raise ValueError("Invalid strategy. Use 'fixed' or 'paragraph'.")

    chunked_docs = []

    for i, chunk in enumerate(text_chunks):
        chunked_docs.append({
            "chunk_id": f"{doc['doc_id']}_chunk_{i}",
            "doc_id": doc["doc_id"],
            "source_type": doc["source_type"],
            "source_name": doc["source_name"],
            "text": chunk,
            "chunk_index": i,
            "chunking_strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "row_number": doc.get("row_number"),
            "page_number": doc.get("page_number")
        })

    return chunked_docs


def chunk_documents(documents: List[Dict], strategy: str = "fixed", chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    all_chunks = []

    for doc in documents:
        if doc.get("source_type") == "pdf":
            doc_strategy = "paragraph" if strategy == "mixed" else strategy
            doc_chunk_size = 1000 if doc_strategy == "paragraph" else chunk_size
            doc_overlap = 200 if doc_strategy == "paragraph" else overlap
        else:
            doc_strategy = "fixed"
            doc_chunk_size = 500
            doc_overlap = 100

        doc_chunks = chunk_document(
            doc,
            strategy=doc_strategy,
            chunk_size=doc_chunk_size,
            overlap=doc_overlap
        )
        all_chunks.extend(doc_chunks)

    return all_chunks