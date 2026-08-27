"""
Hybrid retrieval: BM25 (lexical) + dense (semantic), fused with Reciprocal
Rank Fusion, then optionally re-ranked with a cross-encoder for final
precision. BM25 matters a lot here -- exact drug names, dosage units, and
ICD-style codes are things dense embeddings routinely blur together.
"""
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from chunking.section_chunker import Chunk
from embedding.embedder import Embedder
from indexing.vector_store import VectorStore

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # swap for a biomedical cross-encoder later


@dataclass
class RetrievalResult:
    chunk: Chunk
    dense_score: float
    bm25_score: float
    fused_score: float
    rerank_score: float | None = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, use_reranker: bool = True):
        self.store = store
        self.embedder = embedder
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in store.chunks])
        self.reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(DEFAULT_RERANKER)
            except Exception as e:
                print(f"  [HybridRetriever] WARNING: could not load reranker ({type(e).__name__}: "
                      f"model host unreachable). Continuing with fused BM25+dense ranking only.")

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 30,
        top_k_bm25: int = 30,
        rrf_k: int = 60,
        final_k: int = 5,
        authority_tier_max: int | None = None,   # e.g. 1 to restrict to regulatory sources only
        source_types: list[str] | None = None,   # e.g. ["drug_label"] for dosage questions
    ) -> list[RetrievalResult]:
        # --- dense leg ---
        qvec = self.embedder.embed_query(query)
        dense_hits = self.store.search(qvec, k=top_k_dense)

        # --- lexical leg ---
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k_bm25]

        # --- reciprocal rank fusion ---
        fused: dict[str, dict] = {}
        for rank, (chunk, score) in enumerate(dense_hits):
            fused.setdefault(chunk.chunk_id, {"chunk": chunk, "dense": 0.0, "bm25": 0.0, "rrf": 0.0})
            fused[chunk.chunk_id]["dense"] = score
            fused[chunk.chunk_id]["rrf"] += 1.0 / (rrf_k + rank + 1)

        for rank, idx in enumerate(bm25_ranked_idx):
            chunk = self.store.chunks[idx]
            fused.setdefault(chunk.chunk_id, {"chunk": chunk, "dense": 0.0, "bm25": 0.0, "rrf": 0.0})
            fused[chunk.chunk_id]["bm25"] = float(bm25_scores[idx])
            fused[chunk.chunk_id]["rrf"] += 1.0 / (rrf_k + rank + 1)

        results = [
            RetrievalResult(
                chunk=v["chunk"], dense_score=v["dense"], bm25_score=v["bm25"], fused_score=v["rrf"]
            )
            for v in fused.values()
        ]

        # --- metadata filtering (authority tier / source type) ---
        if authority_tier_max is not None:
            results = [r for r in results if r.chunk.authority_tier <= authority_tier_max]
        if source_types is not None:
            results = [r for r in results if r.chunk.source_type in source_types]

        results.sort(key=lambda r: -r.fused_score)
        candidates = results[: max(final_k * 4, 20)]  # widen before rerank

        # --- cross-encoder re-rank for final precision ---
        if self.reranker and candidates:
            pairs = [(query, r.chunk.text) for r in candidates]
            scores = self.reranker.predict(pairs)
            for r, s in zip(candidates, scores):
                r.rerank_score = float(s)
            candidates.sort(key=lambda r: -(r.rerank_score or 0))

        return candidates[:final_k]
