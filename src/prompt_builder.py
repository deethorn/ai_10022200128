# src/prompt_builder.py
# Student: Chizota Diamond Chizzy
# Index Number: 10022200128

def format_context(chunks: list, max_chunks: int = 2, max_characters: int = 1200) -> str:
    """
    Select and format the best chunks into one context block.
    Keep context short for small local models.
    """
    selected_chunks = chunks[:max_chunks]

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(selected_chunks, start=1):
        source_label = f"Source {i}"
        source_type = chunk.get("source_type", "unknown")
        source_name = chunk.get("source_name", "unknown")

        location = ""
        if chunk.get("page_number") is not None:
            location = f", page {chunk['page_number']}"
        elif chunk.get("row_number") is not None:
            location = f", row {chunk['row_number']}"

        chunk_text = chunk["text"].strip()

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
    """
    Build a grounded RAG prompt with stronger hallucination control.
    """
    context = format_context(
        chunks=retrieved_chunks,
        max_chunks=max_chunks,
        max_characters=max_characters
    )

    prompt = f"""
You are an academic RAG assistant.

Answer the question using ONLY the context below.

Important rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is present in the context, copy the exact fact in your own words.
- If the answer is not present, say: I could not find the answer in the provided documents.
- Give only a short direct answer.

Question:
{query}

Context:
{context}

Direct Answer:
""".strip()

    return prompt


def build_baseline_prompt(query: str) -> str:
    """
    Build a simple baseline prompt without retrieved context.
    """
    prompt = f"""
Answer the following question clearly and briefly.

Question:
{query}

Answer:
""".strip()

    return prompt