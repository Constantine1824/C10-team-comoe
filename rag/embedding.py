"""Text embeddings: sentence-transformers when available, TF-IDF fallback.

Both implementations expose the same interface — ``fit(corpus)``,
``encode_passages(texts)`` and ``encode_queries(texts)`` — and return
L2-normalized dense float32 arrays, so cosine similarity is a plain dot
product everywhere downstream. The fallback exists so retrieval still works
on machines without model weights or network access; ranking quality in that
mode is weaker but the pipeline stays testable.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge-*-en-v1.5 retrieval works best when the query (not the passages) is
# prefixed with this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    """Dense embeddings from a Hugging Face sentence-transformers model."""

    kind = "dense"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def fit(self, corpus: list[str]) -> None:  # nothing to fit; kept for interface parity
        return

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_passages([f"{BGE_QUERY_PREFIX}{t}" for t in texts])


class TfidfEmbedder:
    """TF-IDF vectors in the same dense/normalized shape as the dense path.

    Must be fitted on the chunk corpus before encoding; queries are projected
    with the corpus vocabulary, so unseen query terms are simply ignored.
    """

    kind = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.dimension = 0

    def fit(self, corpus: list[str]) -> None:
        self.vectorizer.fit(corpus)
        self.dimension = len(self.vectorizer.vocabulary_)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise RuntimeError("TfidfEmbedder must be fitted on the corpus before encoding")
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


def load_embedder(
    preference: str = "auto", model_name: str = DEFAULT_MODEL_NAME
) -> SentenceTransformerEmbedder | TfidfEmbedder:
    """Return a dense embedder, or the TF-IDF fallback if requested/unavailable.

    ``preference``: ``"dense"`` (fail loudly if the model can't load),
    ``"tfidf"``, or ``"auto"`` (try dense, fall back with a warning).
    """
    if preference == "tfidf":
        return TfidfEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as error:
        if preference == "dense":
            raise
        print(f"WARNING: dense embedder unavailable ({error}); falling back to TF-IDF")
        return TfidfEmbedder()
