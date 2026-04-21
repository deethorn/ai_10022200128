import re
from collections import Counter


STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "of", "to", "for", "in",
    "on", "at", "by", "with", "and", "or", "did", "does", "do", "what",
    "who", "how", "when", "where", "all", "about", "tell", "me", "say",
    "says", "said", "does", "statement"
}


GENERIC_QUERY_TERMS = {
    "ghana", "budget", "statement", "economic", "policy", "government",
    "financial", "year", "election", "results", "presidential", "general",
    "document", "pdf", "2020", "2024", "2025"
}


BUDGET_FRONT_MATTER_PATTERNS = [
    "the budget statement and economic policy",
    "presented to parliament by",
    "on the authority of",
    "public financial management act",
    "section 1 introduction",
    "table of contents",
    "available for public access",
    "to purchase a physical copy",
    "public relations office",
    "electronic copies can be"
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list:
    return [word for word in normalize_text(text).split() if word not in STOPWORDS]


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


def extract_focus_terms(query: str) -> list:
    focus_terms = []
    for token in tokenize(query):
        if token not in GENERIC_QUERY_TERMS:
            focus_terms.append(token)
    return list(dict.fromkeys(focus_terms))


def specific_focus_score(query: str, text: str) -> float:
    focus_terms = extract_focus_terms(query)
    if not focus_terms:
        return 0.0

    text_words = set(tokenize(text))
    matches = sum(1 for term in focus_terms if term in text_words)
    return matches / len(focus_terms)


def infer_query_domain(query: str) -> str:
    q = normalize_text(query)

    if "budget" in q:
        return "budget"
    if "election" in q or "president" in q or "candidate" in q or "votes" in q:
        return "election"
    return "general"


def infer_query_intent(query: str) -> str:
    q = normalize_text(query)

    if "theme" in q:
        return "theme"
    if "title" in q or "name of" in q:
        return "title"
    if "who won" in q or "winner" in q:
        return "winner"
    if "what does" in q and "say about" in q:
        return "topic_specific"
    if "about" in q:
        return "topic_specific"
    return "general"


def is_budget_front_matter(text: str) -> bool:
    text_lower = normalize_text(text)
    return any(pattern in text_lower for pattern in BUDGET_FRONT_MATTER_PATTERNS)


def score_chunk_for_context(query: str, chunk: dict) -> float:
    text = chunk.get("text", "")
    existing_score = chunk.get("final_score", 0.0)
    keyword_score = keyword_overlap_score(query, text)
    focus_score = specific_focus_score(query, text)

    score = (existing_score * 0.60) + (keyword_score * 0.15) + (focus_score * 0.25)

    domain = infer_query_domain(query)
    intent = infer_query_intent(query)

    if domain == "budget" and intent not in {"theme", "title"} and is_budget_front_matter(text):
        score -= 0.25

    return score


def choose_context_chunks(chunks: list, query: str, max_chunks: int, max_characters: int) -> list:
    if not chunks:
        return []

    scored_chunks = []
    for chunk in chunks:
        item = chunk.copy()
        item["_context_score"] = score_chunk_for_context(query, chunk)
        scored_chunks.append(item)

    scored_chunks.sort(key=lambda x: x["_context_score"], reverse=True)

    selected = []
    current_length = 0

    for chunk in scored_chunks:
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text:
            continue

        source_type = chunk.get("source_type", "unknown")
        source_name = chunk.get("source_name", "unknown")

        location = ""
        if chunk.get("page_number") is not None:
            location = f", page {chunk['page_number']}"
        elif chunk.get("row_number") is not None:
            location = f", row {chunk['row_number']}"

        block = f"[{source_type}: {source_name}{location}]\n{chunk_text}\n"

        if current_length + len(block) > max_characters:
            continue

        selected.append(chunk)
        current_length += len(block)

        if len(selected) >= max_chunks:
            break

    return selected


def format_context(retrieved_chunks: list, max_chunks: int = 2, max_characters: int = 1200, query: str = "") -> str:
    selected_chunks = choose_context_chunks(
        chunks=retrieved_chunks,
        query=query,
        max_chunks=max_chunks,
        max_characters=max_characters
    ) if query else retrieved_chunks[:max_chunks]

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
        retrieved_chunks=retrieved_chunks,
        max_chunks=2,
        max_characters=1200,
        query=query
    )

    if not context.strip():
        context = "No reliable context was retrieved."

    domain = infer_query_domain(query)
    intent = infer_query_intent(query)
    focus_terms = extract_focus_terms(query)
    focus_terms_text = ", ".join(focus_terms) if focus_terms else "None"

    domain_rules = ""

    if domain == "budget":
        domain_rules = """
Budget-specific rules:
- If the question is about a topic or sector, use only context that directly discusses that topic or sector.
- Ignore cover pages, document titles, ceremonial text, introductory headings, and generic front matter unless the question asks for the title, theme, presenter, or date.
- Never answer a sector question with the document title.
- Prefer policy actions, fiscal risks, measures, reforms, allocations, or sector-specific statements.
""".strip()

    elif domain == "election":
        domain_rules = """
Election-specific rules:
- Prefer chunks containing candidate names, party names, vote counts, percentages, regions, or direct result statements.
- If a winner is asked, answer with the winning candidate and party if present in the context.
- Do not add background information that is not shown in the context.
""".strip()

    prompt = f"""
You are an academic RAG assistant.

Answer the question using ONLY the context below.

Core rules:
- Do not use outside knowledge.
- Do not guess.
- First identify which source block most directly answers the question.
- Use the most relevant evidence, not the most general text.
- If the context contains only generic document headers, titles, or introductions and does not contain the actual answer, say exactly: I could not find the answer in the provided documents.
- Keep the answer brief, direct, and factual.
- For topic-specific questions, answer only from lines that directly discuss that topic.
- Do not repeat the document title unless the user asked for the title or theme.

Query domain: {domain}
Query intent: {intent}
Focus terms: {focus_terms_text}

{domain_rules}

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