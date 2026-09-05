"""Lightweight metadata-filtered retrieval for the benchmark documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "for", "how", "in", "is",
    "it", "of", "on", "or", "should", "the", "to", "what", "when",
    "with",
}


def load_documents(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load source documents regardless of the caller's current directory."""
    csv_path = Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "documents.csv"
    frame = pd.read_csv(csv_path).fillna("")
    return frame.to_dict(orient="records")


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text).lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _candidate_documents(
    question: str,
    topic: str,
    care_setting: str,
    population: str,
    documents: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    documents = list(documents)
    exact = [
        document
        for document in documents
        if document.get("topic") == topic
        and document.get("care_setting") == care_setting
        and document.get("population") == population
    ]
    if exact:
        return exact

    topic_matches = [document for document in documents if document.get("topic") == topic]
    return topic_matches or documents


def retrieve_document(
    question: str,
    topic: str,
    care_setting: str,
    population: str,
    documents: Iterable[dict[str, str]],
) -> dict[str, str]:
    """Return the most relevant document using metadata and lexical overlap."""
    candidates = _candidate_documents(
        question, topic, care_setting, population, documents
    )
    question_tokens = _tokens(question)

    def rank(document: dict[str, str]) -> tuple[int, int, str]:
        searchable_text = " ".join(
            str(document.get(field, ""))
            for field in ("title", "text")
        )
        overlap = len(question_tokens & _tokens(searchable_text))
        title_overlap = len(question_tokens & _tokens(str(document.get("title", ""))))
        return overlap, title_overlap, str(document.get("document_id", ""))

    return max(candidates, key=rank)