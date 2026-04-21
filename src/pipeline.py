import re
from src.retriever import retrieve_top_k
from src.prompt_builder import build_rag_prompt, format_context
from src.structured_qa import answer_structured_query


GENERIC_BUDGET_PATTERNS = [
    "available for public access",
    "to purchase a physical copy",
    "public relations office",
    "electronic copies can be"
]

BUDGET_TITLE_PATTERNS = [
    "resetting the economy for the ghana we want",
    "2025 budget statement",
    "budget statement and economic policy",
    "presented to parliament",
]

LLM_FAILURE_PHRASES = [
    "i could not find",
    "could not find the answer",
    "not in the context",
    "no information",
    "i don't know",
    "i do not know",
]


def is_generic_budget_chunk(text: str) -> bool:
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in GENERIC_BUDGET_PATTERNS)


def extract_budget_theme(retrieved_chunks: list):
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        match = re.search(
            r"(Resetting The Economy For The Ghana We Want|Resetting the Economy for the Ghana We Want)",
            text,
            re.IGNORECASE
        )
        if match:
            return match.group(1)
    return None


def extract_sentences_from_context(selected_context: str, min_length: int = 40) -> str:
    lines = selected_context.splitlines()
    sentences = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Source ") and "[" in line:
            continue
        line_lower = line.lower()
        if any(pattern in line_lower for pattern in BUDGET_TITLE_PATTERNS):
            continue
        if len(line) >= min_length:
            sentences.append(line)

    if sentences:
        return " ".join(sentences[:3])

    return ""


def is_llm_failure(answer: str) -> bool:
    if not answer or answer.strip() == "":
        return True
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in LLM_FAILURE_PHRASES)


def extract_theme_answer(query: str, retrieved_chunks: list):
    if "theme" in query.lower():
        return extract_budget_theme(retrieved_chunks)
    return None


def build_debug_fields(query: str, chunks: list):
    selected_context = format_context(
        retrieved_chunks=chunks,
        max_chunks=2,
        max_characters=1200,
        query=query
    )

    if not selected_context.strip():
        selected_context = "No reliable context was retrieved."

    final_prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=chunks,
        max_chunks=2,
        max_characters=1200
    )

    return selected_context, final_prompt


def run_rag_pipeline(query: str, embedder, chunk_embeddings, chunk_docs, llm, top_k: int = 5):
    retrieved_chunks = retrieve_top_k(
        query=query,
        embedder=embedder,
        chunk_embeddings=chunk_embeddings,
        chunk_docs=chunk_docs,
        top_k=top_k,
        use_expansion=True
    )

    first_result = retrieved_chunks[0] if retrieved_chunks else {}

    selected_context, final_prompt = build_debug_fields(query, retrieved_chunks)

    if first_result.get("status") == "needs_clarification":
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "selected_context": selected_context,
            "final_prompt": final_prompt,
            "final_answer": first_result["message"],
            "answer_source": "clarification_needed"
        }

    if first_result.get("status") == "low_confidence":
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "selected_context": selected_context,
            "final_prompt": final_prompt,
            "final_answer": first_result["message"],
            "answer_source": "low_confidence_retrieval"
        }

    answer_chunks = retrieved_chunks

    if "budget" in query.lower():
        filtered_budget_chunks = [
            chunk for chunk in retrieved_chunks
            if not is_generic_budget_chunk(chunk.get("text", ""))
        ]

        if filtered_budget_chunks:
            answer_chunks = filtered_budget_chunks
        else:
            answer_chunks = []

    display_chunks = answer_chunks if answer_chunks else retrieved_chunks
    selected_context, final_prompt = build_debug_fields(query, display_chunks)

    structured = answer_structured_query(query)
    if structured:
        return {
            "query": query,
            "retrieved_chunks": display_chunks,
            "selected_context": selected_context,
            "final_prompt": final_prompt,
            "final_answer": structured["answer"],
            "answer_source": structured["source"]
        }

    if "budget" in query.lower() and not answer_chunks:
        return {
            "query": query,
            "retrieved_chunks": display_chunks,
            "selected_context": selected_context,
            "final_prompt": final_prompt,
            "final_answer": "I could not find a specific answer in the provided budget context.",
            "answer_source": "generic_budget_only"
        }

    first_answer_chunk = answer_chunks[0] if answer_chunks else {}
    top_score = first_answer_chunk.get("final_score", 0.0)

    if top_score < 0.45:
        return {
            "query": query,
            "retrieved_chunks": display_chunks,
            "selected_context": selected_context,
            "final_prompt": final_prompt,
            "final_answer": "I could not find enough relevant evidence in the provided documents.",
            "answer_source": "low_confidence_retrieval"
        }

    # --- theme queries: extractive is more reliable than LLM ---
    theme_answer = extract_theme_answer(query, answer_chunks)
    if theme_answer:
        return {
            "query": query,
            "retrieved_chunks": display_chunks,
            "selected_context": selected_context,
            "final_prompt": final_prompt,
            "final_answer": theme_answer,
            "answer_source": "extractive_theme"
        }

    # --- all other queries: LLM first ---
    final_answer = llm.generate_answer(final_prompt)
    answer_source = "local_llm"

    # --- extractive fallback if LLM fails ---
    if is_llm_failure(final_answer):
        fallback = extract_sentences_from_context(selected_context)
        if fallback:
            final_answer = fallback
            answer_source = "extractive_fallback"
        else:
            final_answer = "I could not find the answer in the provided documents."
            answer_source = "safe_fallback"

    return {
        "query": query,
        "retrieved_chunks": display_chunks,
        "selected_context": selected_context,
        "final_prompt": final_prompt,
        "final_answer": final_answer,
        "answer_source": answer_source
    }