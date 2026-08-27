"""
FAISS-backed dense vector store. Chosen over an external DB for the
prototyping stage per the plan -- swap for Qdrant/pgvector once you need
production-grade metadata filtering, updates, or multi-user access.
"""
import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from chunking.section_chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim, since embeddings are normalized
        self.chunks: list[Chunk] = []  # row i in self.index <-> self.chunks[i]

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]):
        assert embeddings.shape[0] == len(chunks)
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[Chunk, float]]:
        scores, idxs = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, dir_path: str):
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        with open(d / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(d / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "count": len(self.chunks)}, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        d = Path(dir_path)
        meta = json.loads((d / "meta.json").read_text())
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(d / "index.faiss"))
        with open(d / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
