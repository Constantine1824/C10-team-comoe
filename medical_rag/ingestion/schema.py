"""
Common schema every loader normalizes into. This is the contract between
ingestion and chunking -- no downstream code should care whether a doc
came from a PDF guideline, a paper, or a structured drug API.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class RawSection:
    """One structural unit of a source document (a header + its body text)."""
    heading: str
    text: str
    level: int = 1          # heading depth, e.g. H1=1, H2=2 ...
    order: int = 0          # position in the doc, for stable ordering
    is_table: bool = False  # tables need different chunking rules


@dataclass
class SourceDocument:
    """A fully parsed source document, before chunking."""
    doc_id: str                      # stable id, e.g. sha1 of source url/path
    title: str
    source_type: str                 # "guideline" | "paper" | "drug_label"
    source_name: str                 # e.g. "NICE", "PubMed", "openFDA"
    url: Optional[str] = None
    authority_tier: int = 2          # 1=regulatory/guideline, 2=peer-reviewed, 3=other
    publication_date: Optional[date] = None
    retrieved_date: Optional[date] = None
    sections: list[RawSection] = field(default_factory=list)
    extra_metadata: dict = field(default_factory=dict)
