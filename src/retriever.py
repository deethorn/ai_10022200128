# Student: Chizota Diamond Chizzy
# Index Number: 10022200128

import numpy as np
from collections import Counter


def cosine_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    """
    Compute cosine similarity manually.
    """
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
    return float(similarity)


def rank_by_cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, chunk_docs: list):
    """
    Rank all chunks manually using cosine similarity.
    """
    scored_results = []

    for i, doc_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)

        result = chunk_docs[i].copy()
        result["cosine_similarity"] = score
        scored_results.append(result)

    scored_results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return scored_results


def query_expansion(query: str) -> list:
    """
    Very simple manual query expansion.
    """
    expanded_queries = [query]

    query_lower = query.lower()

    if "budget" in query_lower:
        expanded_queries.extend([
            "budget statement",
            "theme of the budget",
            "2025 budget statement Ghana"
        ])

    if "election" in query_lower:
        expanded_queries.extend([
            "ghana election results",
            "presidential election results",
            "votes by candidate"
        ])

    if "theme" in query_lower:
        expanded_queries.extend([
            "title",
            "main theme",
            "budget theme"
        ])

    return list(dict.fromkeys(expanded_queries))


def keyword_overlap_score(query: str, text: str) -> float:
    """
    Compute a simple keyword overlap score.
    """
    query_words = query.lower().split()
    text_words = text.lower().split()

    if not query_words:
        return 0.0

    query_counter = Counter(query_words)
    text_counter = Counter(text_words)

    overlap = 0
    for word in query_counter:
        overlap += min(query_counter[word], text_counter.get(word, 0))

    return overlap / len(query_words)


def hybrid_retrieve(query: str, query_embedding: np.ndarray, chunk_embeddings: np.ndarray, chunk_docs: list, top_k: int = 5):
    """
    Manual hybrid retrieval using cosine similarity + keyword overlap.
    """
    scored_results = []

    for i, doc_embedding in enumerate(chunk_embeddings):
        cosine_score = cosine_similarity(query_embedding, doc_embedding)
        keyword_score = keyword_overlap_score(query, chunk_docs[i]["text"])

        # weighted combination
        final_score = (0.8 * cosine_score) + (0.2 * keyword_score)

        result = chunk_docs[i].copy()
        result["cosine_similarity"] = cosine_score
        result["keyword_overlap"] = keyword_score
        result["final_score"] = final_score
        scored_results.append(result)

    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_results[:top_k]


def retrieve_top_k(query: str, embedder, chunk_embeddings: np.ndarray, chunk_docs: list, top_k: int = 5, use_expansion: bool = True):
    """
    Full retrieval function with optional query expansion.
    """
    if use_expansion:
        expanded_queries = query_expansion(query)
    else:
        expanded_queries = [query]

    all_results = []

    for expanded_query in expanded_queries:
        query_embedding = embedder.embed_query(expanded_query)
        results = hybrid_retrieve(
            query=expanded_query,
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunk_docs=chunk_docs,
            top_k=top_k
        )
        all_results.extend(results)

    # deduplicate by chunk_id, keep best score
    best_results = {}

    for result in all_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in best_results or result["final_score"] > best_results[chunk_id]["final_score"]:
            best_results[chunk_id] = result

    final_results = list(best_results.values())
    final_results.sort(key=lambda x: x["final_score"], reverse=True)

    return final_results[:top_k]