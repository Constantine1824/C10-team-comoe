"""Generate a deterministic nearest-question submission for the benchmark."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _tfidf_cosine(query: str, candidates: pd.Series) -> list[float]:
    documents = [_tokens(text) for text in candidates]
    query_tokens = _tokens(query)
    document_frequency = Counter(token for document in documents for token in set(document))
    query_counts = Counter(query_tokens)
    vocabulary = set(query_counts)
    for document in documents:
        vocabulary.update(document)

    def weight(count: int, document_count: int) -> float:
        if not count:
            return 0.0
        return (1.0 + math.log(count)) * math.log(
            (1 + len(documents)) / (1 + document_frequency[document_count])
        ) + 0.0

    query_vector = {
        token: weight(query_counts[token], token) for token in vocabulary
    }
    scores = []
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    for document in documents:
        counts = Counter(document)
        vector = {token: weight(counts[token], token) for token in vocabulary}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        dot = sum(query_vector[token] * vector[token] for token in vocabulary)
        scores.append(dot / (query_norm * norm) if query_norm and norm else 0.0)
    return scores


def select_candidates(train: pd.DataFrame, question: pd.Series) -> pd.DataFrame:
    exact = train[
        (train["topic"] == question["topic"])
        & (train["care_setting"] == question["care_setting"])
        & (train["population"] == question["population"])
    ]
    if not exact.empty:
        return exact
    topic_matches = train[train["topic"] == question["topic"]]
    return topic_matches if not topic_matches.empty else train


def generate_submission(
    train_path: str | Path = ROOT / "data" / "train_qa.csv",
    test_path: str | Path = ROOT / "data" / "test_questions.csv",
    output_path: str | Path = ROOT / "submissions" / "retrieval_baseline.csv",
) -> pd.DataFrame:
    train = pd.read_csv(train_path).fillna("")
    test = pd.read_csv(test_path).fillna("")
    rows: list[dict[str, object]] = []

    for _, question in test.iterrows():
        candidates = select_candidates(train, question)
        scores = _tfidf_cosine(question["question"], candidates["question"])
        best_index = max(range(len(scores)), key=scores.__getitem__)
        answer = candidates.iloc[best_index]["reference_answer"]
        rows.append({"QuestionId": question["QuestionId"], "Answer": answer})

    submission = pd.DataFrame(rows, columns=["QuestionId", "Answer"])
    submission.to_csv(output_path, index=False)
    return submission


if __name__ == "__main__":
    result = generate_submission()
    print(result.to_string(index=False))