"""
Loader for PDF guidelines and papers.

Medical PDFs are inconsistent (guideline PDFs often have real bookmarks/TOC;
papers usually don't). We try structured extraction first (bookmarks/font-size
heuristics) and fall back to a simple page-based split. Tables are extracted
separately and flagged so the chunker never splits them mid-row.
"""
import hashlib
from datetime import date
from pathlib import Path

import pdfplumber

from .schema import SourceDocument, RawSection


def _hash_id(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:16]


def _looks_like_heading(line: str) -> bool:
    """Cheap heuristic: short line, title-cased or all-caps, no trailing period."""
    line = line.strip()
    if not line or len(line) > 80:
        return False
    if line.endswith((".", ",", ";")):
        return False
    words = line.split()
    if len(words) > 10:
        return False
    caps_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
    return caps_ratio > 0.6


def load_pdf(
    path: str,
    source_type: str,          # "guideline" | "paper"
    source_name: str,
    authority_tier: int,
    title: str | None = None,
    url: str | None = None,
    publication_date: date | None = None,
) -> SourceDocument:
    p = Path(path)
    sections: list[RawSection] = []
    order = 0
    current_heading = "Introduction"
    current_lines: list[str] = []

    def flush():
        nonlocal order, current_heading, current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(RawSection(heading=current_heading, text=text, order=order))
            order += 1
        current_lines = []

    with pdfplumber.open(p) as pdf:
        for page in pdf.pages:
            # Extract tables first and flag them as their own sections,
            # so they never get merged/split with surrounding prose.
            for tbl in page.extract_tables():
                rows = ["\t".join(cell or "" for cell in row) for row in tbl]
                if rows:
                    sections.append(RawSection(
                        heading=f"{current_heading} (table)",
                        text="\n".join(rows),
                        order=order,
                        is_table=True,
                    ))
                    order += 1

            text = page.extract_text() or ""
            for line in text.split("\n"):
                if _looks_like_heading(line):
                    flush()
                    current_heading = line.strip()
                else:
                    current_lines.append(line)
    flush()

    return SourceDocument(
        doc_id=_hash_id(str(p.resolve())),
        title=title or p.stem,
        source_type=source_type,
        source_name=source_name,
        url=url,
        authority_tier=authority_tier,
        publication_date=publication_date,
        retrieved_date=date.today(),
        sections=sections,
    )
