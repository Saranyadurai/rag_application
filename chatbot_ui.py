"""
Streamlit chatbot UI for the ClinicalTrials.gov protocol RAG app.

Usage:
    $env:FIREWORKS_API_KEY = "fw_your_key_here"
    streamlit run chatbot_ui.py

This wraps the existing LangGraph pipeline from app.py directly -- no
separate backend needed. The retrieval pipeline (BM25 index, Chroma store,
cross-encoder reranker) and the graph are built once and cached across
interactions using Streamlit's resource cache, so each question only pays
the cost of retrieval + generation, not of re-loading everything from disk.

Place this file in the same folder as app.py and retrieve.py.
"""

import os
import streamlit as st

from app import build_app

st.set_page_config(page_title="Clinical Trial Protocol Assistant", page_icon="🧪", layout="centered")


@st.cache_resource(show_spinner=False)
def get_app():
    return build_app(top_k=6)


st.title("🧪 Clinical Trial Protocol Assistant")
st.caption(
    "Ask questions about trial design, endpoints, and eligibility criteria across "
    "24 Alzheimer's/MCI/dementia clinical trial protocols. Answers are grounded "
    "in retrieved protocol excerpts and cite the source trial and page."
)

if not os.environ.get("FIREWORKS_API_KEY"):
    st.error(
        "FIREWORKS_API_KEY environment variable not set. Set it before launching:\n\n"
        '`$env:FIREWORKS_API_KEY = "fw_your_key_here"` (PowerShell)\n\n'
        "then restart this app."
    )
    st.stop()

with st.spinner("Loading retrieval pipeline (BM25 index, vector store, reranker)... this happens once."):
    rag_app = get_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render prior turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Example questions to help people get started
with st.sidebar:
    st.subheader("Example questions")
    examples = [
        "What is the primary endpoint of NCT03531710?",
        "What are the key inclusion criteria for NCT01300728?",
        "Compare the primary endpoints of NCT03531710 and NCT05189106.",
        "What is the primary endpoint of NCT02051608?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_query = ex

    st.divider()
    st.caption(
        "Corpus: 24 ClinicalTrials.gov protocol PDFs (Alzheimer's/MCI/dementia). "
        "1 trial (NCT02051608) was excluded due to a source-PDF extraction failure -- "
        "the assistant will correctly refuse questions about it."
    )

# Handle either a typed question or a clicked example
query = st.chat_input("Ask a question about a trial...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = rag_app.invoke({"question": query, "docs": [], "answer": ""})
            answer = result["answer"]
            st.markdown(answer)

            # Show the retrieved sources in an expander for transparency
            if result.get("docs"):
                with st.expander(f"Sources ({len(result['docs'])} retrieved chunks)"):
                    for doc, score in result["docs"]:
                        meta = doc.metadata
                        st.markdown(
                            f"**{meta['nct_id']}** | {meta['section']} | p.{meta['page']} | score={score:.2f}"
                        )
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
