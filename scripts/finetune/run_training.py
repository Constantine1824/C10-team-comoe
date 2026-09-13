"""Train the adapter and write a grounded benchmark submission.

Training pairs use the gold document's chunks as context; test questions get
context from the hybrid RAG retriever (dense + lexical, RRF-fused). Both go
through the same formatter so the model sees identically shaped context at
train and inference time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset

from rag.retriever import HybridRetriever
from utils.format import format_test_data, format_train_data
from trainer import evaluate, finetune


ROOT = Path(__file__).resolve().parents[1]


def prepare_data(
    retriever: HybridRetriever, retrieval_k: int = 3
) -> tuple[Dataset, Dataset, pd.DataFrame]:
    train_frame = pd.read_csv(ROOT / "data" / "train_qa.csv").fillna("")
    # gold context: the labeled document, chunked and formatted exactly like
    # retrieved context will be at inference time
    train_frame["context"] = [
        retriever.context_for_document(document_id)
        for document_id in train_frame["document_id"]
    ]

    test_frame = pd.read_csv(ROOT / "data" / "test_questions.csv").fillna("")
    test_frame["context"] = [
        retriever.retrieve_context(
            row["question"],
            row["topic"],
            row["care_setting"],
            row["population"],
            k=retrieval_k,
        )
        for _, row in test_frame.iterrows()
    ]

    train_data = Dataset.from_pandas(train_frame, preserve_index=False).map(format_train_data)
    test_data = Dataset.from_pandas(test_frame, preserve_index=False).map(format_test_data)
    return train_data, test_data, test_frame


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith("Answer:"):
        answer = answer[len("Answer:"):].strip()
    return answer


def main(output_path: str | Path = ROOT / "submissions" / "finetuned_submission.csv") -> None:
    retriever = HybridRetriever()
    train_data, test_data, test_frame = prepare_data(retriever)
    trainer = finetune(train_data)
    answers = [clean_answer(answer) for answer in evaluate(trainer, test_data, batch_size=8)]
    if len(answers) != len(test_frame):
        raise RuntimeError("The model did not return one answer per test question")

    submission = pd.DataFrame({"QuestionId": test_frame["QuestionId"], "Answer": answers})
    if submission["Answer"].str.strip().eq("").any():
        raise RuntimeError("Generated submission contains a blank answer")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} answers to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "submissions" / "finetuned_submission.csv"))
    args = parser.parse_args()
    main(args.output)
