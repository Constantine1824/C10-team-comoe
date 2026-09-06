"""Hybrid retrieval over document chunks: dense + lexical, fused with RRF.

Candidate chunks are first restricted by the question's metadata (exact
topic/care_setting/population match, falling back to topic match, then the
whole corpus — same ladder as the document-level baseline in
``rag.retrieval``). Within candidates, dense vector search and lexical TF-IDF
matching each produce a ranking, fused with Reciprocal Rank Fusion so no
blend weight has to be hand-tuned: exact terms like drug or disease names
that embeddings blur are recovered by the lexical leg.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_SENTENCES,
    chunk_documents,
    format_chunk_body,
    load_documents,
)
from .embedding import load_embedder
from .indexing import VectorStore

RRF_K = 60
DEFAULT_TOP_K = 3
CANDIDATE_POOL = 10


class HybridRetriever:
    def __init__(
        self,
        documents: list[dict[str, str]] | None = None,
        embedder=None,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
        index_dir: str | None = None,
    ) -> None:
        self.documents = documents if documents is not None else load_documents()

        self.embedder = embedder if embedder is not None else load_embedder()
        if index_dir and (Path(index_dir) / "chunks.json").exists():
            self.store = VectorStore.load(index_dir)
            self.chunks = self.store.chunks
        else:
            self.chunks = chunk_documents(self.documents, max_chars, overlap_sentences)
            corpus = [format_chunk_body(chunk) for chunk in self.chunks]
            self.embedder.fit(corpus)
            self.store = VectorStore(self.embedder.encode_passages(corpus), self.chunks)
            if index_dir:
                self.store.save(index_dir)

        corpus = [format_chunk_body(chunk) for chunk in self.chunks]
        self.lexical = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.lexical_matrix = self.lexical.fit_transform(corpus)

    # -- candidate selection -------------------------------------------------

    def _candidate_chunks(
        self, topic: str, care_setting: str, population: str
    ) -> set[str]:
        def matching(predicate) -> set[str]:
            return {
                chunk["chunk_id"]
                for chunk in self.chunks
                if predicate(chunk) and chunk["text"].strip()
            }

        exact = matching(
            lambda c: c["topic"] == topic
            and c["care_setting"] == care_setting
            and c["population"] == population
        )
        if exact:
            return exact
        by_topic = matching(lambda c: c["topic"] == topic)
        return by_topic or {chunk["chunk_id"] for chunk in self.chunks}

    # -- ranking -------------------------------------------------------------

    def _dense_ranking(
        self, query: str, allowed: set[str], n: int
    ) -> list[str]:
        query_vector = self.embedder.encode_queries([query])[0]
        return [
            chunk["chunk_id"]
            for _, chunk in self.store.search(query_vector, k=n, allowed_chunk_ids=allowed)
        ]

    def _lexical_ranking(self, query: str, allowed: set[str], n: int) -> list[str]:
        scores = np.asarray(
            (self.lexical_matrix @ self.lexical.transform([query]).T).todense()
        ).ravel()
        order = np.argsort(-scores)
        ranked = [
            self.chunks[index]["chunk_id"]
            for index in order
            if scores[index] > 0 and self.chunks[index]["chunk_id"] in allowed
        ]
        return ranked[:n]

    def retrieve(
        self,
        query: str,
        topic: str = "",
        care_setting: str = "",
        population: str = "",
        k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Top-k chunks for a question, fused across dense and lexical legs."""
        allowed = self._candidate_chunks(topic, care_setting, population)
        dense = self._dense_ranking(query, allowed, CANDIDATE_POOL)
        lexical = self._lexical_ranking(query, allowed, CANDIDATE_POOL)

        fused: dict[str, float] = {}
        for rank, chunk_id in enumerate(dense, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, chunk_id in enumerate(lexical, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:k]
        results = []
        for chunk_id in ranked_ids:
            chunk = dict(by_id[chunk_id])
            chunk["score"] = fused[chunk_id]
            chunk["dense_rank"] = dense.index(chunk_id) + 1 if chunk_id in dense else None
            chunk["lexical_rank"] = (
                lexical.index(chunk_id) + 1 if chunk_id in lexical else None
            )
            results.append(chunk)
        return results

    # -- prompt-facing helpers ----------------------------------------------

    def retrieve_context(
        self,
        query: str,
        topic: str = "",
        care_setting: str = "",
        population: str = "",
        k: int = DEFAULT_TOP_K,
        separator: str = "\n\n",
    ) -> str:
        """Retrieved chunks joined into one context string for a model prompt."""
        chunks = self.retrieve(query, topic, care_setting, population, k)
        return separator.join(format_chunk_body(chunk) for chunk in chunks)

    def context_for_document(self, document_id: str, separator: str = "\n\n") -> str:
        """All chunks of one document, formatted exactly like retrieved context.

        Used for training pairs so the model sees the same context shape at
        train time (gold document) and inference time (retrieved chunks).
        """
        chunks = [chunk for chunk in self.chunks if chunk["document_id"] == document_id]
        if not chunks:
            raise KeyError(f"unknown document_id: {document_id}")
        return separator.join(format_chunk_body(chunk) for chunk in chunks)


@lru_cache(maxsize=None)
def get_retriever(embedder_preference: str = "auto") -> HybridRetriever:
    """Process-wide retriever; building it downloads/loads the embedder once."""
    return HybridRetriever(embedder=load_embedder(embedder_preference))
