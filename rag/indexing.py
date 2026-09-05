"""Vector store over chunk embeddings: cosine search plus save/load.

Embeddings arrive L2-normalized from the embedder, so cosine similarity is a
matrix-vector product. With benchmark-sized corpora (tens to hundreds of
chunks) a flat numpy index is exact and dependency-free; swap in FAISS or a
vector database only when scale demands it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and chunks ({len(chunks)}) must align"
            )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.chunks = list(chunks)

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[tuple[float, dict]]:
        """Top-k chunks by cosine similarity, optionally restricted to candidates."""
        scores = self.embeddings @ np.asarray(query_embedding, dtype=np.float32)
        if allowed_chunk_ids is not None:
            mask = np.array(
                [chunk["chunk_id"] in allowed_chunk_ids for chunk in self.chunks]
            )
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [
            (float(scores[index]), self.chunks[index])
            for index in order
            if np.isfinite(scores[index])
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        embeddings = np.load(directory / "embeddings.npy")
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls(embeddings, chunks)
