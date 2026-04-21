import re


def clean_text(text: str, source_type: str = "unknown") -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\xa0", " ")

    if source_type == "pdf":
        text = text.replace("\t", " ")
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r" *([,.;:!?])", r"\1", text)
        text = re.sub(r"\n +", "\n", text)
    else:
        text = text.replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


def clean_document(doc: dict) -> dict:
    cleaned_doc = doc.copy()
    cleaned_doc["text"] = clean_text(
        doc.get("text", ""),
        source_type=doc.get("source_type", "unknown")
    )
    return cleaned_doc


def clean_documents(documents: list) -> list:
    cleaned = []

    for doc in documents:
        cleaned_doc = clean_document(doc)
        if cleaned_doc["text"]:
            cleaned.append(cleaned_doc)

    return cleaned