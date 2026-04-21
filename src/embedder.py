from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import EMBEDDING_MODEL

class TextEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list) -> np.ndarray:
        """
        Convert a list of texts into embeddings.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert one user query into an embedding.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding[0]