"""Split source documents into metadata-enriched chunks for embedding.

Each chunk carries its document metadata, and the text that gets embedded
has that metadata front-loaded so the vector captures topic/setting/population
context, not just the body. Short factsheets (like this benchmark's) pass
through as a single chunk; longer documents are packed sentence-by-sentence
into windows with a carried-over sentence for overlap.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

DEFAULT_MAX_CHARS = 600
DEFAULT_OVERLAP_SENTENCES = 1

METADATA_FIELDS = ("document_id", "title", "topic", "care_setting", "population")


def load_documents(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load source documents regardless of the caller's current directory."""
    csv_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[1] / "data" / "documents.csv"
    )
    frame = pd.read_csv(csv_path).fillna("")
    return frame.to_dict(orient="records")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(str(text)) if s.strip()]


def split_into_chunks(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[str]:
    """Pack sentences into windows of at most ``max_chars`` characters.

    The last ``overlap_sentences`` sentences of a full window carry over to
    start the next one, so a clinical rule that straddles a boundary stays
    readable in both chunks. A single sentence longer than ``max_chars`` is
    emitted as its own oversized chunk rather than cut mid-sentence.
    """
    sentences = _sentences(text)
    if not sentences:
        return [str(text).strip()] if str(text).strip() else []

    windows: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > max_chars:
            windows.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) + 1 for s in current) - 1 if current else 0
        current.append(sentence)
        current_len += len(sentence) + (1 if current_len else 0)
    if current:
        windows.append(current)

    return [" ".join(window) for window in windows]


def format_chunk_body(chunk: dict[str, str]) -> str:
    """Metadata header + chunk text; this is the string that gets embedded."""
    header = (
        f"Title: {chunk.get('title', '')}\n"
        f"Topic: {chunk.get('topic', '')} | Care setting: {chunk.get('care_setting', '')} "
        f"| Population: {chunk.get('population', '')}"
    )
    return f"{header}\n{chunk.get('text', '')}"


def chunk_documents(
    documents: Iterable[dict[str, str]],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[dict[str, str]]:
    """Chunk every document, returning one dict per chunk with metadata."""
    chunks: list[dict[str, str]] = []
    for document in documents:
        pieces = split_into_chunks(document.get("text", ""), max_chars, overlap_sentences)
        for index, piece in enumerate(pieces):
            chunk = {field: str(document.get(field, "")) for field in METADATA_FIELDS}
            chunk["chunk_index"] = index
            chunk["num_chunks"] = len(pieces)
            chunk["chunk_id"] = f"{document.get('document_id', 'doc')}#c{index:02d}"
            chunk["text"] = piece
            chunks.append(chunk)
    return chunks


def chunk_documents_from_csv(
    path: str | Path | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[dict[str, str]]:
    return chunk_documents(load_documents(path), max_chars, overlap_sentences)
