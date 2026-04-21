def format_context(chunks: list, max_chunks: int = 2, max_characters: int = 1200) -> str:
    selected_chunks = chunks[:max_chunks]

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(selected_chunks, start=1):
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text:
            continue

        source_label = f"Source {i}"
        source_type = chunk.get("source_type", "unknown")
        source_name = chunk.get("source_name", "unknown")

        location = ""
        if chunk.get("page_number") is not None:
            location = f", page {chunk['page_number']}"
        elif chunk.get("row_number") is not None:
            location = f", row {chunk['row_number']}"

        block = (
            f"{source_label} [{source_type}: {source_name}{location}]\n"
            f"{chunk_text}\n"
        )

        if current_length + len(block) > max_characters:
            break

        context_parts.append(block)
        current_length += len(block)

    return "\n".join(context_parts)


def build_rag_prompt(query: str, retrieved_chunks: list, max_chunks: int = 2, max_characters: int = 1200) -> str:
    context = format_context(
        chunks=retrieved_chunks,
        max_chunks=max_chunks,
        max_characters=max_characters
    )

    if not context.strip():
        context = "No reliable context was retrieved."

    prompt = f"""
You are an academic RAG assistant.

Answer the question using ONLY the context below.

Important rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is present in the context, answer briefly and directly.
- If the answer is not present, say exactly: I could not find the answer in the provided documents.
- Do not explain beyond the evidence.

Question:
{query}

Context:
{context}

Direct Answer:
""".strip()

    return prompt


def build_baseline_prompt(query: str) -> str:
    prompt = f"""
Answer the following question clearly and briefly.

Question:
{query}

Answer:
""".strip()

    return prompt