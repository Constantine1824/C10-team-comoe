"""
Chunking strategy:
  - Never split a table section (dosage tables, interaction matrices).
  - Keep whole sections together when they fit the token budget.
  - Split oversized sections on paragraph boundaries, not fixed windows,
    so we don't cut a sentence like "...contraindicated in patients with"
    right before the condition that matters.
  - Merge consecutive undersized sections (e.g. short guideline subsections)
    so retrieval doesn't return fragments with no context.

Token counting uses a cheap whitespace heuristic by default; swap in a real
tokenizer (tiktoken / the embedding model's tokenizer) for production.
"""
from dataclasses import dataclass, field
from typing import Optional

from ingestion.schema import SourceDocument, RawSection

MIN_CHUNK_TOKENS = 40
MAX_CHUNK_TOKENS = 350
MERGE_TARGET_TOKENS = 200


def _count_tokens(text: str) -> int:
    return len(text.split())


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_type: str
    source_name: str
    authority_tier: int
    heading: str
    text: str
    is_table: bool
    order: int
    url: Optional[str] = None
    extra_metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        """Flat metadata dict for the vector store payload."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "authority_tier": self.authority_tier,
            "heading": self.heading,
            "is_table": self.is_table,
            "url": self.url,
            **self.extra_metadata,
        }


def _split_oversized(section: RawSection, max_tokens: int) -> list[str]:
    """Split on paragraph boundaries; only fall back to sentence splits if a
    single paragraph is still oversized."""
    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [section.text]

    pieces, current, current_tokens = [], [], 0
    for para in paragraphs:
        ptoks = _count_tokens(para)
        if ptoks > max_tokens:
            # paragraph itself too big -> sentence split as last resort
            if current:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            sentences = para.replace("\n", " ").split(". ")
            buf, buf_toks = [], 0
            for s in sentences:
                st = _count_tokens(s)
                if buf_toks + st > max_tokens and buf:
                    pieces.append(". ".join(buf) + ".")
                    buf, buf_toks = [], 0
                buf.append(s)
                buf_toks += st
            if buf:
                pieces.append(". ".join(buf))
            continue

        if current_tokens + ptoks > max_tokens and current:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += ptoks

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(doc: SourceDocument) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending_heading, pending_text, pending_tokens = None, [], 0

    def flush_pending():
        nonlocal pending_heading, pending_text, pending_tokens
        if pending_text:
            text = "\n\n".join(pending_text)
            chunks.append(_make_chunk(doc, pending_heading, text, len(chunks), is_table=False))
        pending_heading, pending_text, pending_tokens = None, [], 0

    # Structured sources (e.g. openFDA drug labels) arrive pre-segmented into
    # semantically distinct, high-precision fields (Dosage, Contraindications,
    # Interactions...). Merging them defeats the purpose of structured data --
    # a retrieval hit on "Dosage" should never drag in "Pregnancy" just because
    # both fields were short. So we never merge sections for these sources,
    # same as tables.
    never_merge = doc.source_type == "drug_label"

    for section in sorted(doc.sections, key=lambda s: s.order):
        toks = _count_tokens(section.text)

        if section.is_table or never_merge:
            flush_pending()
            chunks.append(_make_chunk(doc, section.heading, section.text, len(chunks), is_table=section.is_table))
            continue

        if toks > MAX_CHUNK_TOKENS:
            flush_pending()
            for piece in _split_oversized(section, MAX_CHUNK_TOKENS):
                chunks.append(_make_chunk(doc, section.heading, piece, len(chunks), is_table=False))
            continue

        if toks < MIN_CHUNK_TOKENS:
            # merge small sections together up to MERGE_TARGET_TOKENS
            if pending_tokens + toks > MERGE_TARGET_TOKENS:
                flush_pending()
            heading = pending_heading or section.heading
            pending_heading = heading if pending_heading is None else f"{pending_heading} / {section.heading}"
            pending_text.append(f"{section.heading}: {section.text}")
            pending_tokens += toks
            continue

        # normal-sized standalone section
        flush_pending()
        chunks.append(_make_chunk(doc, section.heading, section.text, len(chunks), is_table=False))

    flush_pending()
    return chunks


def _make_chunk(doc: SourceDocument, heading: str, text: str, idx: int, is_table: bool) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}::{idx}",
        doc_id=doc.doc_id,
        doc_title=doc.title,
        source_type=doc.source_type,
        source_name=doc.source_name,
        authority_tier=doc.authority_tier,
        heading=heading or "",
        text=text,
        is_table=is_table,
        order=idx,
        url=doc.url,
        extra_metadata=doc.extra_metadata,
    )
