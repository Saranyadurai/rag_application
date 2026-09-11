"""
LangGraph RAG app for ClinicalTrials.gov protocol Q&A.

Flow: retrieve -> generate (with explicit refusal path) -> END

Usage:
    $env:FIREWORKS_API_KEY = "fw_your_key_here"
    python app.py --query "What is the primary endpoint of NCT03531710?"

    # or interactive mode (no --query):
    python app.py

Design notes:
  - The refusal path is enforced in TWO places, not left to the LLM alone:
      1. A hard score-threshold check on the reranked results -- if nothing
         retrieved clears a minimum relevance bar, we refuse WITHOUT calling
         the LLM at all (cheaper, and guarantees refusal can't be talked out
         of by a confidently-worded question).
      2. A prompt instruction telling the LLM to say so explicitly if the
         provided context doesn't actually answer the question, even when
         retrieval did return something on-topic but incomplete.
  - Every answer must cite [NCT_ID p.PAGE] for every claim, sourced directly
    from chunk metadata -- not invented by the model.
"""

import argparse
import os

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from retrieve import build_retrieval_pipeline, retrieve_and_rerank

# NOTE: An earlier version of this used a hard absolute score threshold
# (MIN_RELEVANCE_SCORE = -4.0) to refuse before calling the LLM. Eval testing
# showed this was a bug, not a safety net: cross-encoder relevance scores are
# NOT on a stable, comparable scale across different queries/documents -- the
# correct answer to "target sample size for NCT01972204" scored around -10
# (because of that document's awkward machine-translated phrasing dragging
# down cross-encoder confidence generally), while an unrelated query scored
# +0.21 as its top result. A fixed cutoff picked from one query's score
# range caused a false refusal on a different query with genuinely correct,
# well-ranked content. Refusal now relies on the LLM reading the actual
# retrieved content (semantically grounded) rather than an arbitrary numeric
# scale. We keep only a bare check for the genuinely-empty case.

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
# Confirmed available via list_fireworks_models.py against this account's
# actual catalog (search-engine results for "current Fireworks models" were
# unreliable/stale -- always verify against /v1/models directly for your
# own account). deepseek-v4-flash is a fast, cheap tier well suited to
# grounded extraction/citation tasks like this RAG app. For stronger
# (slower, pricier) answers, try glm-5p3 or deepseek-v4-pro-0813 instead:
#   $env:GENERATION_MODEL = "accounts/fireworks/models/glm-5p3"
DEFAULT_MODEL = os.environ.get(
    "GENERATION_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731"
)

SYSTEM_PROMPT = """You are a clinical trial design assistant. Answer ONLY using the provided context, which consists of excerpts from clinical trial protocol documents.

Rules:
1. Every factual claim you make MUST be followed by a citation in the format [NCT_ID p.PAGE], taken directly from the excerpt it came from.
2. If the context does not contain enough information to answer the question, say so explicitly: "I could not find this in the retrieved protocols." Do not guess or use outside knowledge.
3. If the question asks about a specific trial that doesn't appear in the context at all, say so explicitly rather than answering about a different trial.
4. Be precise with numbers, dosages, and criteria -- do not paraphrase specific eligibility criteria or statistical parameters in ways that change their meaning.
5. If different excerpts appear to conflict, note the conflict rather than silently picking one.
"""


class RAGState(TypedDict):
    question: str
    docs: List[Tuple[Document, float]]
    answer: str


def build_app(top_k: int = 6):
    bm25, vectorstore, reranker, all_docs = build_retrieval_pipeline()
    llm = ChatOpenAI(
        model=DEFAULT_MODEL,
        base_url=FIREWORKS_BASE_URL,
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        temperature=0,
    )

    def retrieve_node(state: RAGState):
        results = retrieve_and_rerank(state["question"], bm25, vectorstore, reranker, all_docs, top_k=top_k)
        return {"docs": results}

    def generate_node(state: RAGState):
        results = state["docs"]

        if not results:
            return {"answer": "I could not find this in the retrieved protocols."}

        context_blocks = []
        for doc, score in results:
            meta = doc.metadata
            context_blocks.append(
                f"[{meta['nct_id']} p.{meta['page']} | {meta['section']}]\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {state['question']}",
            },
        ]

        response = llm.invoke(messages)
        return {"answer": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Query the clinical trial protocol RAG app.")
    parser.add_argument("--query", help="Question to ask. If omitted, starts interactive mode.")
    parser.add_argument("--top_k", type=int, default=6)
    args = parser.parse_args()

    if not os.environ.get("FIREWORKS_API_KEY"):
        raise RuntimeError(
            "FIREWORKS_API_KEY environment variable not set. "
            'Set it with: $env:FIREWORKS_API_KEY = "fw_your_key_here"'
        )

    print("Building retrieval pipeline and app graph (loads BM25, Chroma, reranker)...\n")
    app = build_app(top_k=args.top_k)

    def ask(question: str):
        result = app.invoke({"question": question, "docs": [], "answer": ""})
        print(f"\nQ: {question}\n")
        print(f"A: {result['answer']}\n")
        print("-" * 60)

    if args.query:
        ask(args.query)
    else:
        print("Interactive mode. Type a question, or 'quit' to exit.\n")
        while True:
            q = input("Q: ").strip()
            if q.lower() in ("quit", "exit"):
                break
            if not q:
                continue
            result = app.invoke({"question": q, "docs": [], "answer": ""})
            print(f"\nA: {result['answer']}\n")


if __name__ == "__main__":
    main()
