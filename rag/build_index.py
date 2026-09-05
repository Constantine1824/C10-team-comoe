"""Build and persist the chunk index: chunk -> embed -> save.

Writes the vector store (embeddings + chunk payloads) under ``rag/index/``.
Subsequent HybridRetriever(index_dir=...) calls load it instead of
re-embedding the corpus. Run from the repo root:
    python -m rag.build_index
"""

from __future__ import annotations

from pathlib import Path

from .retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = ROOT / "rag" / "index"


def main(index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
    retriever = HybridRetriever(index_dir=str(index_dir))
    print(f"documents: {len(retriever.documents)}")
    print(f"chunks indexed: {len(retriever.chunks)}")
    print(f"embedder: {retriever.embedder.kind} "
          f"({getattr(retriever.embedder, 'model_name', 'n/a')}), "
          f"dim={retriever.store.dimension}")
    print(f"index saved to: {index_dir}")


if __name__ == "__main__":
    main()
