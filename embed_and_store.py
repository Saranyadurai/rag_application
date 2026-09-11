"""
Embed chunked protocol text and store in a persistent Chroma vector store.

Usage:
    python embed_and_store.py --chunks_file ./chunks/chunks.jsonl --persist_dir ./ctgov_chroma

Uses a LOCAL sentence-transformers model (default: BAAI/bge-base-en-v1.5) --
no API key needed, runs entirely on your CPU, completely free. The model
downloads once (a few hundred MB) on first run and is cached locally after
that.

To use a different local model, set the EMBEDDING_MODEL environment variable
to any sentence-transformers-compatible model name, e.g.:
    $env:EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
"""

import argparse
import json
import os
import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def load_chunks(chunks_file: str, excluded_ids=None):
    excluded_ids = excluded_ids or set()
    chunks = []
    skipped = 0
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("nct_id") in excluded_ids:
                skipped += 1
                continue
            chunks.append(c)
    if skipped:
        print(f"Skipped {skipped} chunk(s) from excluded document(s): {sorted(excluded_ids)}")
    return chunks


def chunks_to_documents(chunks):
    docs = []
    for c in chunks:
        docs.append(
            Document(
                page_content=c["text"],
                metadata={
                    "nct_id": c["nct_id"],
                    "section": c["section"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                    "token_count": c["token_count"],
                },
            )
        )
    return docs


def build_embeddings_client():
    model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    print(f"Loading local embedding model: {model} (this may take a minute on first run while it downloads)...")
    # normalize_embeddings=True is recommended for BGE-family models -- it
    # makes cosine similarity behave correctly for retrieval.
    return HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_and_store(docs, persist_dir: str, batch_size: int = 100, max_retries: int = 3):
    embeddings = build_embeddings_client()
    vectorstore = None
    total = len(docs)

    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        attempt = 0
        while True:
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        batch, embeddings, persist_directory=persist_dir
                    )
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = 2 ** attempt
                print(f"  Batch {i}-{i+len(batch)} failed ({e}); retrying in {wait}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait)

        done = min(i + batch_size, total)
        print(f"Embedded {done}/{total} chunks...")

    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into a persistent Chroma vector store.")
    parser.add_argument("--chunks_file", default="./chunks/chunks.jsonl")
    parser.add_argument("--persist_dir", default="./ctgov_chroma")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated NCT IDs to skip, as a safety net in case chunks_file still "
        "contains them from an earlier run. Example: --exclude NCT02051608",
    )
    args = parser.parse_args()

    excluded_ids = {x.strip() for x in args.exclude.split(",") if x.strip()}

    print(f"Loading chunks from {args.chunks_file}...")
    chunks = load_chunks(args.chunks_file, excluded_ids=excluded_ids)
    print(f"Loaded {len(chunks)} chunks.")

    docs = chunks_to_documents(chunks)

    start = time.time()
    vectorstore = embed_and_store(docs, args.persist_dir, args.batch_size)
    elapsed = time.time() - start

    print(f"\nDone. Vector store persisted to {args.persist_dir}/")
    print(f"Total chunks embedded: {len(docs)}")
    print(f"Elapsed time: {elapsed:.1f}s")

    # quick sanity check: run one test query
    print("\nRunning a sanity-check query: 'primary endpoint'...")
    results = vectorstore.similarity_search("primary endpoint", k=3)
    for r in results:
        print(f"  [{r.metadata['nct_id']} | {r.metadata['section']} | p.{r.metadata['page']}] {r.page_content[:100]}...")


if __name__ == "__main__":
    main()
