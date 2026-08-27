# Medical SLM — RAG Component

Retrieval pipeline for a domain-specific medical SLM, feeding grounded
context to the fine-tuned generation model. Built in dependency order:

```
ingestion → chunking → embedding/indexing → hybrid retrieval → (eval)
```

## Layout

```
ingestion/
  schema.py            common SourceDocument/RawSection format for every loader
  pdf_loader.py         guidelines & papers (PDF, heading-aware, table-safe)
  structured_loader.py  drug data (openFDA label JSON), pre-segmented fields

chunking/
  section_chunker.py    section-aware, table-safe, merges undersized sections
                         (never merges structured drug-label fields — each
                         field like "Dosage" or "Contraindications" stays
                         its own chunk, since blending them defeats the
                         point of structured data)

embedding/
  embedder.py            sentence-transformers wrapper (BAAI/bge-small-en-v1.5
                          by default); falls back to a hashed bag-of-words
                          vector if the model host is unreachable, purely so
                          the rest of the pipeline can still be smoke-tested

indexing/
  vector_store.py        FAISS flat index + metadata payload, save/load

retrieval/
  hybrid_retriever.py    BM25 (lexical) + dense (semantic), fused with
                          Reciprocal Rank Fusion, then cross-encoder
                          re-ranked; supports authority_tier / source_type
                          filtering (e.g. restrict dosage Qs to drug_label
                          + guideline tier-1 sources)

eval/
  retrieval_eval.py      recall@k / MRR harness against a hand-labeled
                          query -> relevant_chunk_ids set

demo.py                  end-to-end run over the sample corpus
data/sample/              openFDA-style sample + a generated guideline PDF
```

## Running the demo

```bash
pip install -r requirements.txt
python demo.py
```

This ingests the sample drug labels + guideline PDF, chunks them, embeds
and indexes them, then runs a few hybrid retrieval queries end to end.

**Note:** if the model host (huggingface.co) isn't reachable from wherever
you run this, `Embedder` and the reranker fall back automatically so you
can still verify the wiring — but ranking quality in that mode is not
representative. Run for real with actual network/model access before
trusting any retrieval quality numbers.

## Design decisions worth knowing about

- **Structured drug data is never merged at chunk time.** Each openFDA
  field (Dosage, Contraindications, Interactions...) becomes its own
  chunk. Early in building this we merged small sections together
  (fine for guideline subsections) and it silently collapsed an entire
  drug label into one blob — a retrieval hit on "starting dose" would've
  dragged in Pregnancy/Contraindications text too. Tables get the same
  treatment for the same reason.
- **Hybrid retrieval, not pure dense.** Drug names, dosage units, and
  guideline codes are exact-match terms that dense embeddings alone
  often blur — BM25 catches what semantic search misses, and RRF fusion
  avoids having to hand-tune a blend weight.
- **authority_tier** (1=regulatory/guideline, 2=peer-reviewed, 3=other)
  is a first-class filter, not just metadata — dosage/safety questions
  should be restricted to tier-1 sources at query time.

## Next steps (not yet built)

1. **Real eval set.** Hand-label 30-50 (query, relevant_chunk_id) pairs
   from your actual corpus, run `eval/retrieval_eval.py`, and use that —
   not vibes — to decide on embedding model choice, chunk size, and RRF
   vs. weighted-sum fusion.
2. **Query rewriting.** Patient phrasing ("my chest hurts when I breathe")
   vs. clinical phrasing ("pleuritic chest pain") — a small rewrite step
   before retrieval will materially help recall.
3. **Groundedness / citation check.** Once this plugs into your
   teammate's fine-tuned model, add a check that flags generated claims
   not traceable to the retrieved chunks, and force the model to say "I
   don't have reliable information on this" rather than fill gaps.
4. **Swap FAISS flat index for Qdrant/pgvector** once you need metadata
   filtering at scale, concurrent writes, or more than a few hundred
   thousand chunks.
5. **Real biomedical embedding/reranker models** (e.g. a PubMedBERT-based
   embedder, a biomedical cross-encoder) — only worth doing after step 1
   gives you a baseline to measure the lift against.
