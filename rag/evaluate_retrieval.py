"""Evaluate retrieval quality against the gold document labels in train_qa.

Each train question lists the ``document_id`` its reference answer came
from, so the training set doubles as a labeled retrieval benchmark. Prints
recall@k (did the gold document appear in the top-k retrieved?) and mean
reciprocal rank; also compares the hybrid retriever against the pure
lexical document-level baseline (``rag.retrieval``) so any lift is visible.

Run from the repo root:
    python -m rag.evaluate_retrieval
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from .retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
METADATA_FIELDS = ("topic", "care_setting", "population")


def load_labeled_questions(
    path: str | Path | None = None,
) -> pd.DataFrame:
    csv_path = path if path else ROOT / "data" / "train_qa.csv"
    frame = pd.read_csv(csv_path).fillna("")
    if "document_id" not in frame.columns:
        raise ValueError("train_qa.csv must contain a document_id column to evaluate against")
    return frame


def evaluate_retriever(
    retriever: HybridRetriever,
    questions: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    hits = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []

    for _, row in questions.iterrows():
        retrieved = retriever.retrieve(
            row["question"], *(row[field] for field in METADATA_FIELDS), k=max(k_values)
        )
        ranks = [
            index + 1
            for index, chunk in enumerate(retrieved)
            if chunk["document_id"] == row["document_id"]
        ]
        rank = ranks[0] if ranks else None
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            for k in k_values:
                if rank <= k:
                    hits[k] += 1

    total = len(questions)
    metrics = {f"recall@{k}": hits[k] / total for k in k_values}
    metrics["mrr"] = sum(reciprocal_ranks) / total if total else 0.0
    metrics["questions"] = float(total)
    return metrics


def evaluate_baseline(questions: pd.DataFrame) -> dict[str, float]:
    """The existing document-level lexical baseline, same metrics."""
    from .retrieval import retrieve_document

    hits = 0
    for _, row in questions.iterrows():
        documents = retriever_documents()
        top = retrieve_document(
            row["question"], *(row[field] for field in METADATA_FIELDS), documents
        )
        if top["document_id"] == row["document_id"]:
            hits += 1
    return {"recall@1": hits / len(questions), "questions": float(len(questions))}


def retriever_documents():
    from .retrieval import load_documents

    return load_documents()


def main() -> None:
    questions = load_labeled_questions()
    retriever = HybridRetriever()

    print(f"corpus: {len(retriever.chunks)} chunks from {len(retriever.documents)} documents")
    print(f"embedder: {retriever.embedder.kind} ({getattr(retriever.embedder, 'model_name', 'n/a')})")
    print(f"questions: {len(questions)}\n")

    metrics = evaluate_retriever(retriever, questions)
    print("hybrid retriever:")
    for name, value in metrics.items():
        if name != "questions":
            print(f"  {name}: {value:.3f}")

    baseline = evaluate_baseline(questions)
    print("\ndocument-level lexical baseline:")
    print(f"  recall@1: {baseline['recall@1']:.3f}")

    misses = []
    for _, row in questions.iterrows():
        retrieved = retriever.retrieve(
            row["question"], *(row[field] for field in METADATA_FIELDS), k=1
        )
        if retrieved and retrieved[0]["document_id"] != row["document_id"]:
            misses.append((row["QuestionId"], row["question"], retrieved[0]["document_id"]))
    if misses:
        print(f"\ntop-1 misses ({len(misses)}):")
        for question_id, question, predicted in misses:
            print(f"  Q{question_id} '{question[:60]}' -> {predicted}")


if __name__ == "__main__":
    main()
