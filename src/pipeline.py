import re
from src.retriever import retrieve_top_k
from src.prompt_builder import build_rag_prompt, format_context
from src.structured_qa import answer_structured_query


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


def extract_direct_answer(query: str, retrieved_chunks: list):
    query_lower = query.lower()

    if "theme" in query_lower:
        theme = extract_budget_theme(retrieved_chunks)
        if theme:
            return theme

    return None


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

    if first_result.get("status") == "needs_clarification":
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "selected_context": "",
            "final_prompt": "",
            "final_answer": first_result["message"],
            "answer_source": "clarification_needed"
        }

    if first_result.get("status") == "low_confidence":
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "selected_context": "",
            "final_prompt": "",
            "final_answer": first_result["message"],
            "answer_source": "low_confidence_retrieval"
        }

    selected_context = format_context(
        chunks=retrieved_chunks,
        max_chunks=2,
        max_characters=1200
    )

    final_prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_chunks=2,
        max_characters=1200
    )

    structured = answer_structured_query(query)
    extracted_answer = extract_direct_answer(query, retrieved_chunks)

    if structured:
        final_answer = structured["answer"]
        answer_source = structured["source"]
    elif extracted_answer:
        final_answer = extracted_answer
        answer_source = "extractive_fallback"
    else:
        final_answer = llm.generate_answer(final_prompt)
        answer_source = "local_llm"

    if not final_answer or final_answer.strip() == "":
        final_answer = "I could not find the answer in the provided documents."
        answer_source = "safe_fallback"

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "selected_context": selected_context,
        "final_prompt": final_prompt,
        "final_answer": final_answer,
        "answer_source": answer_source
    }