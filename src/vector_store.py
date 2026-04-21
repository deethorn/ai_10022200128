import faiss
import numpy as np
import pickle
from pathlib import Path

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def add_embeddings(self, embeddings: np.ndarray, chunk_docs: list):
        """
        Add embeddings and matching chunk metadata to the FAISS index.
        """
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(chunk_docs)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search the FAISS index and return top-k chunks with distances.
        """
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            result = self.metadata[idx].copy()
            result["faiss_distance"] = float(distances[0][rank])
            result["rank"] = rank + 1
            results.append(result)

        return results

    def save(self, index_path: str, metadata_path: str):
        """
        Save FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str):
        """
        Load FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)