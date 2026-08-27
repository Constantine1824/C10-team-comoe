"""
Retrieval evaluation harness.

Build a small hand-labeled set (start with 30-50, grow to 150-200) of
(query, relevant_chunk_ids) pairs -- ideally written by someone with domain
knowledge, not scraped. Track recall@k and MRR as you change chunking,
embedding models, or fusion weights, so improvements are measured, not vibes.

Example eval_set.json:
[
  {"query": "starting dose of metformin", "relevant_chunk_ids": ["<chunk_id>"]},
  ...
]
"""
import json
from dataclasses import dataclass

from retrieval.hybrid_retriever import HybridRetriever


@dataclass
class EvalCase:
    query: str
    relevant_chunk_ids: list[str]


def load_eval_set(path: str) -> list[EvalCase]:
    data = json.loads(open(path).read())
    return [EvalCase(**d) for d in data]


def evaluate(retriever: HybridRetriever, eval_set: list[EvalCase], k_values=(1, 3, 5, 10)) -> dict:
    recall_hits = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for case in eval_set:
        results = retriever.retrieve(case.query, final_k=max(k_values))
        retrieved_ids = [r.chunk.chunk_id for r in results]

        # recall@k: did we get at least one relevant chunk in the top k?
        for k in k_values:
            if any(cid in retrieved_ids[:k] for cid in case.relevant_chunk_ids):
                recall_hits[k] += 1

        # reciprocal rank: 1 / position of first relevant chunk
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in case.relevant_chunk_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(eval_set)
    return {
        **{f"recall@{k}": recall_hits[k] / n for k in k_values},
        "mrr": sum(reciprocal_ranks) / n,
        "n_cases": n,
    }


if __name__ == "__main__":
    import argparse
    from embedding.embedder import Embedder
    from indexing.vector_store import VectorStore

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--eval-set", required=True)
    args = parser.parse_args()

    store = VectorStore.load(args.index_dir)
    embedder = Embedder()
    retriever = HybridRetriever(store, embedder)
    eval_set = load_eval_set(args.eval_set)

    metrics = evaluate(retriever, eval_set)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
