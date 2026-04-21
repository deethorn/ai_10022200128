import re

def clean_text(text: str) -> str:
    """
    Clean raw text from CSV rows and PDF pages.
    """
    if text is None:
        return ""

    text = str(text)

    # Replace non-breaking spaces with normal spaces
    text = text.replace("\xa0", " ")

    # Replace newlines and tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ")

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    # Remove space before punctuation like "word ,"
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Final trim
    text = text.strip()

    return text


def clean_document(doc: dict) -> dict:
    """
    Clean one document dictionary and return the updated version.
    """
    cleaned_doc = doc.copy()
    cleaned_doc["text"] = clean_text(doc.get("text", ""))
    return cleaned_doc


def clean_documents(documents: list) -> list:
    """
    Clean a list of document dictionaries.
    """
    cleaned = []

    for doc in documents:
        cleaned_doc = clean_document(doc)

        if cleaned_doc["text"]:
            cleaned.append(cleaned_doc)

    return cleaned