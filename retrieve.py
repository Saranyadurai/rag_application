"""
Hybrid retrieval (BM25 + dense) with local cross-encoder reranking, over the
ClinicalTrials.gov protocol corpus.

Usage (interactive test):
    python retrieve.py --query "What is the primary endpoint?" --top_k 6

Also importable as a module by the LangGraph app (step 8):
    from retrieve import build_retrieval_pipeline, retrieve_and_rerank
    bm25, dense, reranker = build_retrieval_pipeline()
    results = retrieve_and_rerank("your question", bm25, dense, reranker, top_k=6)

Pipeline:
  1. BM25 retriever (k=10) -- catches exact matches: drug names, NCT IDs,
     dosage numbers, endpoint terminology.
  2. Dense retriever over the Chroma store (k=10) -- catches semantic intent
     even when wording differs from the protocol text.
  3. Reciprocal Rank Fusion combines both (weights: 0.4 BM25 / 0.6 dense),
     implemented manually here rather than via LangChain's EnsembleRetriever
     to avoid depending on a class whose import path has moved across
     LangChain versions.
  4. A local cross-encoder (free, no API key) reranks the combined candidates
     and the query together, producing a relevance score per chunk. Top_k
     after reranking is what actually goes to the generator in step 8.
"""

import argparse
import json
import os
import re

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

NCT_ID_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def reciprocal_rank_fusion(ranked_lists, weights, k: int = 60):
    """
    Combine multiple ranked lists of Documents into one, using Reciprocal
    Rank Fusion (RRF) -- the same algorithm LangChain's EnsembleRetriever
    uses internally. Implemented directly here to avoid depending on
    EnsembleRetriever, whose import location has moved across LangChain
    versions.

    score(doc) = sum over retrievers of weight / (k + rank_in_that_list)

    Documents are deduplicated by (nct_id, chunk_index) since the same chunk
    can be retrieved by both BM25 and dense search.
    """
    scores = {}
    doc_lookup = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked_list, start=1):
            key = (doc.metadata.get("nct_id"), doc.metadata.get("chunk_index"))
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            doc_lookup[key] = doc

    ranked_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
    return [doc_lookup[k_] for k_ in ranked_keys]

CHUNKS_FILE = "./chunks/chunks.jsonl"
PERSIST_DIR = "./ctgov_chroma"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_documents(chunks_file: str = CHUNKS_FILE):
    docs = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            docs.append(
                Document(
                    page_content=c["text"],
                    metadata={
                        "nct_id": c["nct_id"],
                        "section": c["section"],
                        "page": c["page"],
                        "chunk_index": c["chunk_index"],
                    },
                )
            )
    return docs


def build_retrieval_pipeline(chunks_file: str = CHUNKS_FILE, persist_dir: str = PERSIST_DIR, dense_k: int = 10, bm25_k: int = 10):
    print("Loading documents for BM25...")
    all_docs = load_documents(chunks_file)

    print("Building BM25 retriever...")
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = bm25_k

    print(f"Loading Chroma vector store from {persist_dir}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print(f"Loading local cross-encoder reranker: {RERANKER_MODEL} (downloads once, then cached)...")
    reranker = CrossEncoder(RERANKER_MODEL)

    return bm25, vectorstore, reranker, all_docs


def retrieve_and_rerank(query: str, bm25, vectorstore, reranker, all_docs, top_k: int = 6,
                         rrf_weights=(0.4, 0.6), dense_k: int = 10, bm25_k: int = 10):
    """
    Run BM25 and dense retrieval, combine with Reciprocal Rank Fusion, then
    rerank with the local cross-encoder. Returns a list of (Document, score)
    tuples sorted by relevance, highest first.

    If the query names one or more specific NCT IDs, retrieval is scoped to
    those trials' chunks via metadata filtering on BOTH retrievers (all IDs
    mentioned, not just the first). This matters because the NCT ID itself
    never appears in the protocol document's own text (it's an identifier
    assigned by the registry, not something the document refers to
    internally) -- so content-based search alone can never match on it, even
    though we know exactly which document(s) the person means.
    """
    nct_ids = list(dict.fromkeys(m.upper() for m in NCT_ID_RE.findall(query)))  # dedupe, preserve order

    if nct_ids:
        # Pull a wider candidate pool per scoped trial than the general
        # (unscoped) case. We're already filtering to one document, so a
        # larger k is cheap -- and it matters: compound, multi-part queries
        # (e.g. comparing two trials at once) dilute the embedding/BM25
        # signal for any one specific subsection (like "exclusion criteria"),
        # so the right chunk can miss a narrow top-10 pool even though it
        # retrieves correctly for simpler single-topic queries. Confirmed via
        # eval testing on NCT01300728's exclusion criteria section.
        scoped_dense_k = max(dense_k, 20)
        scoped_bm25_k = max(bm25_k, 20)
        bm25_results = []
        dense_results = []
        for nct_id in nct_ids:
            dense_results.extend(
                vectorstore.similarity_search(query, k=scoped_dense_k, filter={"nct_id": nct_id})
            )
            scoped_docs = [d for d in all_docs if d.metadata.get("nct_id") == nct_id]
            if scoped_docs:
                scoped_bm25 = BM25Retriever.from_documents(scoped_docs)
                scoped_bm25.k = scoped_bm25_k
                bm25_results.extend(scoped_bm25.invoke(query))
    else:
        bm25_results = bm25.invoke(query)
        dense_results = vectorstore.as_retriever(search_kwargs={"k": dense_k}).invoke(query)

    candidates = reciprocal_rank_fusion([bm25_results, dense_results], weights=rrf_weights)
    if not candidates:
        return []

    if len(nct_ids) > 1:
        # Guarantee balanced representation across all mentioned trials.
        # Without this, a global rerank-then-cut can let one trial's chunks
        # score systematically higher (e.g. because the query names that
        # trial's drug explicitly) and crowd the other trial out of the
        # final top_k entirely, even when its content was correctly
        # retrieved and genuinely relevant. Confirmed via eval testing:
        # NCT05189106 content ranked 14th/15th behind six NCT03531710
        # chunks for a two-trial comparison question, so a flat top_k=6
        # cut returned zero content for the second trial.
        per_id_k = max(1, top_k // len(nct_ids))
        final_results = []
        for nct_id in nct_ids:
            id_candidates = [d for d in candidates if d.metadata.get("nct_id") == nct_id]
            if not id_candidates:
                continue
            pairs = [(query, d.page_content) for d in id_candidates]
            scores = reranker.predict(pairs)
            scored_id = sorted(zip(id_candidates, scores), key=lambda x: x[1], reverse=True)
            final_results.extend(scored_id[:per_id_k])
        return final_results

    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Test hybrid retrieval + reranking with a query.")
    parser.add_argument("--query", required=True, help="Question to test retrieval with")
    parser.add_argument("--top_k", type=int, default=6)
    args = parser.parse_args()

    bm25, vectorstore, reranker, all_docs = build_retrieval_pipeline()

    print(f"\nQuery: {args.query}\n")
    results = retrieve_and_rerank(args.query, bm25, vectorstore, reranker, all_docs, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    print(f"Top {len(results)} reranked results:\n")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"[{i}] score={score:.3f} | {doc.metadata['nct_id']} | {doc.metadata['section']} | p.{doc.metadata['page']}")
        print(f"    {doc.page_content[:200]}...")
        print()


if __name__ == "__main__":
    main()
