from typing import List, Dict

def fixed_size_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into fixed-size character chunks with overlap.
    """
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


def paragraph_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text by paragraph-like blocks first, then merge them into chunks.
    If a block is too large, fall back to fixed-size chunking.
    """
    if not text:
        return []

    blocks = [block.strip() for block in text.split("  ") if block.strip()]

    if len(blocks) == 1:
        # fallback if no clear paragraph blocks exist
        blocks = [block.strip() for block in text.split(". ") if block.strip()]
        blocks = [block + "." if not block.endswith(".") else block for block in blocks]

    chunks = []
    current_chunk = ""

    for block in blocks:
        candidate = f"{current_chunk} {block}".strip()

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            if len(block) > chunk_size:
                split_blocks = fixed_size_chunk_text(block, chunk_size, overlap)
                chunks.extend(split_blocks)
                current_chunk = ""
            else:
                current_chunk = block

    if current_chunk:
        chunks.append(current_chunk)

    return add_overlap_to_chunks(chunks, overlap)


def add_overlap_to_chunks(chunks: List[str], overlap: int = 100) -> List[str]:
    """
    Add backward overlap between neighboring chunks.
    """
    if not chunks:
        return []

    overlapped_chunks = [chunks[0]]

    for i in range(1, len(chunks)):
        previous_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        overlap_text = previous_chunk[-overlap:] if len(previous_chunk) > overlap else previous_chunk
        merged_chunk = f"{overlap_text} {current_chunk}".strip()
        overlapped_chunks.append(merged_chunk)

    return overlapped_chunks


def chunk_document(doc: Dict, strategy: str = "fixed", chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Chunk a single document and preserve metadata.
    """
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
    """
    Chunk a list of documents.
    """
    all_chunks = []

    for doc in documents:
        doc_chunks = chunk_document(doc, strategy, chunk_size, overlap)
        all_chunks.extend(doc_chunks)

    return all_chunks