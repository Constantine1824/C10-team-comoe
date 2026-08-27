"""
End-to-end smoke test of the RAG pipeline:
  ingestion -> chunking -> embedding -> indexing -> hybrid retrieval

Run: python demo.py
"""
from ingestion.pdf_loader import load_pdf
from ingestion.structured_loader import load_openfda_file
from chunking.section_chunker import chunk_document
from embedding.embedder import Embedder
from indexing.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever


def main():
    print("=== 1. INGESTION ===")
    drug_docs = load_openfda_file("data/sample/sample_openfda.json")
    guideline_doc = load_pdf(
        "data/sample/sample_guideline.pdf",
        source_type="guideline",
        source_name="Sample T2D Guideline",
        authority_tier=1,
        title="Management of Type 2 Diabetes in Adults",
    )
    all_docs = drug_docs + [guideline_doc]
    for d in all_docs:
        print(f"  loaded '{d.title}' ({d.source_type}, {len(d.sections)} sections)")

    print("\n=== 2. CHUNKING ===")
    all_chunks = []
    for d in all_docs:
        chunks = chunk_document(d)
        all_chunks.extend(chunks)
        print(f"  '{d.title}' -> {len(chunks)} chunks")
    print(f"  total: {len(all_chunks)} chunks")

    print("\n=== 3. EMBEDDING + INDEXING ===")
    embedder = Embedder()
    vecs = embedder.embed_texts([c.text for c in all_chunks])
    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, all_chunks)
    print(f"  indexed {len(all_chunks)} chunks, dim={vecs.shape[1]}")

    print("\n=== 4. HYBRID RETRIEVAL ===")
    retriever = HybridRetriever(store, embedder, use_reranker=True)

    queries = [
        "What is the starting dose of metformin?",
        "Can I take metformin if my kidneys aren't working well?",
        "What antibiotic should I avoid if I'm allergic to penicillin?",
        "When should metformin be paused before a contrast CT scan?",
    ]

    for q in queries:
        print(f"\n  Q: {q}")
        results = retriever.retrieve(q, final_k=3)
        for r in results:
            preview = r.chunk.text.replace("\n", " ")[:110]
            rerank_str = f"{r.rerank_score:.2f}" if r.rerank_score is not None else "n/a"
            print(
                f"    [{r.chunk.source_type:11s} | tier {r.chunk.authority_tier} | "
                f"fused={r.fused_score:.3f} | rerank={rerank_str}] {r.chunk.doc_title} :: {r.chunk.heading}"
            )
            print(f"        {preview}...")


if __name__ == "__main__":
    main()
