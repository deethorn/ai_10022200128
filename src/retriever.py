import re
import numpy as np
from collections import Counter


STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "of", "to", "for", "in",
    "on", "at", "by", "with", "and", "or", "did", "does", "do", "what",
    "who", "how", "when", "where", "all", "about", "tell"
}

AMBIGUOUS_SHORT_QUERIES = {
    "who won",
    "what happened",
    "what is it",
    "tell me more",
    "who is the winner"
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list:
    text = normalize_text(text)
    return [word for word in text.split() if word not in STOPWORDS]


def is_ambiguous_query(query: str) -> bool:
    q = normalize_text(query)
    if q in AMBIGUOUS_SHORT_QUERIES:
        return True
    if len(q.split()) <= 2:
        return True
    return False


def cosine_similarity(query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
    return float(max(0.0, similarity))


def rank_by_cosine_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, chunk_docs: list):
    scored_results = []

    for i, doc_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)

        result = chunk_docs[i].copy()
        result["chunk_id"] = result.get("chunk_id", f"chunk_{i}")
        result["cosine_similarity"] = score
        scored_results.append(result)

    scored_results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return scored_results


def query_expansion(query: str) -> list:
    expanded_queries = [query]
    query_lower = normalize_text(query)

    if "budget" in query_lower and "2025" in query_lower:
        expanded_queries.extend([
            "2025 budget statement ghana",
            f"{query} ghana"
        ])

    if "election" in query_lower and "ghana" in query_lower:
        expanded_queries.extend([
            "ghana presidential election results",
            f"{query} presidential results"
        ])

    if "theme" in query_lower and "budget" in query_lower:
        expanded_queries.extend([
            "theme of the 2025 budget statement"
        ])

    return list(dict.fromkeys(expanded_queries))


def keyword_overlap_score(query: str, text: str) -> float:
    query_words = tokenize(query)
    text_words = tokenize(text)

    if not query_words:
        return 0.0

    query_counter = Counter(query_words)
    text_counter = Counter(text_words)

    overlap = 0
    for word in query_counter:
        overlap += min(query_counter[word], text_counter.get(word, 0))

    return overlap / len(query_words)


def hybrid_retrieve(
    query: str,
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_docs: list,
    top_k: int = 5,
    alpha: float = 0.8
):
    scored_results = []

    for i, doc_embedding in enumerate(chunk_embeddings):
        cosine_score = cosine_similarity(query_embedding, doc_embedding)
        keyword_score = keyword_overlap_score(query, chunk_docs[i].get("text", ""))

        final_score = (alpha * cosine_score) + ((1 - alpha) * keyword_score)

        result = chunk_docs[i].copy()
        result["chunk_id"] = result.get("chunk_id", f"chunk_{i}")
        result["cosine_similarity"] = cosine_score
        result["keyword_overlap"] = keyword_score
        result["final_score"] = final_score
        scored_results.append(result)

    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_results[:top_k]


def retrieve_top_k(
    query: str,
    embedder,
    chunk_embeddings: np.ndarray,
    chunk_docs: list,
    top_k: int = 5,
    use_expansion: bool = True,
    score_threshold: float = 0.35
):
    if is_ambiguous_query(query):
        return [{
            "chunk_id": "clarification_needed",
            "text": "",
            "cosine_similarity": 0.0,
            "keyword_overlap": 0.0,
            "final_score": 0.0,
            "status": "needs_clarification",
            "message": "Your question is ambiguous. Please mention whether you mean the Ghana election or the 2025 budget."
        }]

    expanded_queries = query_expansion(query) if use_expansion else [query]
    all_results = []

    for idx, expanded_query in enumerate(expanded_queries):
        query_embedding = embedder.embed_query(expanded_query)
        results = hybrid_retrieve(
            query=expanded_query,
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunk_docs=chunk_docs,
            top_k=max(top_k, 8)
        )

        for result in results:
            item = result.copy()
            item["expanded_query"] = expanded_query
            item["expansion_rank"] = idx

            if idx == 0:
                item["final_score"] += 0.03

            all_results.append(item)

    best_results = {}

    for result in all_results:
        chunk_id = result["chunk_id"]
        if chunk_id not in best_results or result["final_score"] > best_results[chunk_id]["final_score"]:
            best_results[chunk_id] = result

    final_results = [
        result for result in best_results.values()
        if result["final_score"] >= score_threshold
    ]

    final_results.sort(key=lambda x: x["final_score"], reverse=True)

    if not final_results:
        return [{
            "chunk_id": "no_good_match",
            "text": "",
            "cosine_similarity": 0.0,
            "keyword_overlap": 0.0,
            "final_score": 0.0,
            "status": "low_confidence",
            "message": "I could not find enough relevant evidence in the provided documents."
        }]

    return final_results[:top_k]