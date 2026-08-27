"""
Thin wrapper around a sentence-transformers model so the rest of the
pipeline doesn't care which embedding model is in use. Start with a strong
general-purpose model; swap in a PubMedBERT/BioLORD-style biomedical
embedder later once you have an eval set to actually measure the lift.
"""
import hashlib
import re

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # good quality/speed tradeoff to start
FALLBACK_DIM = 384


class Embedder:
    """
    Wraps a sentence-transformers model. If the model can't be downloaded
    (no network to huggingface.co -- e.g. this sandbox), falls back to a
    deterministic hashed bag-of-words vector so the *rest of the pipeline*
    (indexing, hybrid fusion, filtering) can still be exercised end-to-end.

    The fallback is NOT semantically meaningful -- it's for wiring checks
    only. Real deployments must run where the model can actually load
    (local weights, or network access to the model host).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.online = True
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"  [Embedder] WARNING: could not load '{model_name}' ({type(e).__name__}: "
                  f"model host unreachable). Falling back to a hashed bag-of-words embedder "
                  f"for wiring/testing only -- NOT semantically meaningful.")
            self.online = False
            self.model = None

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(FALLBACK_DIM, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % FALLBACK_DIM] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.online:
            return self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return np.stack([self._hash_embed(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        if self.online:
            prefixed = f"Represent this query for retrieving relevant medical documents: {query}"
            return self.model.encode([prefixed], normalize_embeddings=True)[0]
        return self._hash_embed(query)
