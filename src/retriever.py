#Name: Chizota Diamond Chizzy
#Index Number: 10022200128

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


GENERIC_QUERY_TERMS = {
    "2020", "2024", "2025",
    "ghana", "budget", "statement", "speech", "government", "financial", "year",
    "election", "results", "presidential", "candidate", "party",
    "say", "says", "said", "about", "theme"
}


GENERIC_BUDGET_PATTERNS = [
    "the budget statement and economic policy",
    "available for public access",
    "to purchase a physical copy",
    "public relations office",
    "electronic copies can be downloaded",
    "table of contents",
    "presented to parliament by",
    "on the authority of"
]


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


def is_generic_budget_chunk(text: str) -> bool:
    text_lower = normalize_text(text)
    return any(pattern in text_lower for pattern in GENERIC_BUDGET_PATTERNS)


def extract_specific_terms(query: str) -> list:
    terms = []
    for token in tokenize(query):
        if token not in GENERIC_QUERY_TERMS:
            terms.append(token)
    return list(dict.fromkeys(terms))


def specific_term_coverage(query: str, text: str) -> float:
    query_terms = extract_specific_terms(query)
    if not query_terms:
        return 0.0

    text_words = set(tokenize(text))
    matched = sum(1 for term in query_terms if term in text_words)
    return matched / len(query_terms)


def query_expansion(query: str) -> list:
    expanded_queries = [query]
    query_lower = normalize_text(query)

    if "budget" in query_lower and "2025" in query_lower:
        if "theme" in query_lower:
            expanded_queries.extend([
                "2025 budget theme ghana",
                "theme of the 2025 budget statement ghana",
                "resetting the economy for the ghana we want"
            ])
        elif "energy" in query_lower:
            expanded_queries.extend([
                "2025 budget energy sector ghana",
                "2025 energy sector measures ghana budget",
                "energy sector recovery programme ghana 2025 budget",
                "energy sector fiscal risks ghana 2025 budget"
            ])
        else:
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
    specific_terms = extract_specific_terms(query)

    for i, doc_embedding in enumerate(chunk_embeddings):
        chunk_text = chunk_docs[i].get("text", "")
        cosine_score = cosine_similarity(query_embedding, doc_embedding)
        keyword_score = keyword_overlap_score(query, chunk_text)
        specific_score = specific_term_coverage(query, chunk_text)

        final_score = (alpha * cosine_score) + ((1 - alpha) * keyword_score)

        if specific_terms:
            final_score += 0.20 * specific_score

            if specific_score == 0:
                final_score -= 0.12

            if "budget" in normalize_text(query) and is_generic_budget_chunk(chunk_text):
                final_score -= 0.18

        result = chunk_docs[i].copy()
        result["chunk_id"] = result.get("chunk_id", f"chunk_{i}")
        result["cosine_similarity"] = cosine_score
        result["keyword_overlap"] = keyword_score
        result["specific_term_coverage"] = specific_score
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

    primary_query = expanded_queries[0]
    primary_embedding = embedder.embed_query(primary_query)
    primary_results = hybrid_retrieve(
        query=primary_query,
        query_embedding=primary_embedding,
        chunk_embeddings=chunk_embeddings,
        chunk_docs=chunk_docs,
        top_k=max(top_k, 12)
    )

    candidate_map = {}

    for result in primary_results:
        item = result.copy()
        item["expanded_query"] = primary_query
        item["expansion_rank"] = 0
        item["primary_score"] = result["final_score"]
        item["expansion_support"] = 0.0
        candidate_map[item["chunk_id"]] = item

    for idx, expanded_query in enumerate(expanded_queries[1:], start=1):
        query_embedding = embedder.embed_query(expanded_query)
        results = hybrid_retrieve(
            query=expanded_query,
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            chunk_docs=chunk_docs,
            top_k=max(top_k, 8)
        )

        for result in results:
            chunk_id = result["chunk_id"]

            if chunk_id in candidate_map:
                candidate_map[chunk_id]["expansion_support"] = max(
                    candidate_map[chunk_id]["expansion_support"],
                    result["final_score"]
                )
            else:
                item = result.copy()
                item["expanded_query"] = expanded_query
                item["expansion_rank"] = idx
                item["primary_score"] = 0.0
                item["expansion_support"] = result["final_score"]
                candidate_map[chunk_id] = item

    final_results = []

    for item in candidate_map.values():
        if item["primary_score"] > 0:
            item["final_score"] = item["primary_score"] + (0.08 * item["expansion_support"])
        else:
            item["final_score"] = item["expansion_support"] * 0.85

        if item["final_score"] >= score_threshold:
            final_results.append(item)

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
