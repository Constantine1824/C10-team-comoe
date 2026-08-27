"""
Loader for structured drug data -- e.g. openFDA drug label JSON
(https://open.fda.gov/apis/drug/label/) or a DrugBank-style export.

Structured sources are the highest-value part of the corpus for factual
questions (dosage, interactions, contraindications) because they arrive
pre-segmented -- we don't need heading heuristics, we trust the field names.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

from .schema import SourceDocument, RawSection

# openFDA label fields we care about, in priority order.
# (full field list: https://open.fda.gov/apis/drug/label/searchable-fields/)
OPENFDA_FIELDS = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "drug_interactions",
    "adverse_reactions",
    "pregnancy",
]


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load_openfda_record(record: dict, source_name: str = "openFDA") -> SourceDocument:
    """
    Parse a single record from an openFDA drug label JSON response
    (i.e. one element of the `results` array).
    """
    brand = (record.get("openfda", {}).get("brand_name") or ["Unknown"])[0]
    generic = (record.get("openfda", {}).get("generic_name") or [""])[0]
    title = f"{brand} ({generic})" if generic else brand

    sections = []
    for order, field in enumerate(OPENFDA_FIELDS):
        values = record.get(field)
        if not values:
            continue
        text = "\n".join(values) if isinstance(values, list) else str(values)
        sections.append(RawSection(
            heading=field.replace("_", " ").title(),
            text=text.strip(),
            order=order,
        ))

    return SourceDocument(
        doc_id=_hash_id(source_name, brand, generic),
        title=title,
        source_type="drug_label",
        source_name=source_name,
        authority_tier=1,  # regulatory source -> highest trust tier
        retrieved_date=date.today(),
        sections=sections,
        extra_metadata={"brand_name": brand, "generic_name": generic},
    )


def load_openfda_file(path: str, source_name: str = "openFDA") -> list[SourceDocument]:
    """Load a full openFDA JSON export (a {"results": [...]} file) into SourceDocuments."""
    data = json.loads(Path(path).read_text())
    records = data.get("results", data if isinstance(data, list) else [])
    return [load_openfda_record(r, source_name) for r in records]
