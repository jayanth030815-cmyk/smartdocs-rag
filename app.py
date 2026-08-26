import os
import shutil
import time
from pathlib import Path
import streamlit as st

# Core pipeline imports
from core.ingestion.document_loader import DocumentLoader
from core.ingestion.chunker import RecursiveCharacterChunker
from core.indexing.dense_index import DenseVectorStore
from core.indexing.sparse_index import SparseBM25Index
from core.generation.llm_client import LLMClient
from core.pipeline import SmartDocsPipeline

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="SmartDocs — Advanced Multi-Strategy RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Session State Initialization
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    dense_store = DenseVectorStore(collection_name="smartdocs_streamlit")
    sparse_index = SparseBM25Index()
    llm_client = LLMClient()
    st.session_state.chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
    st.session_state.dense_store = dense_store
    st.session_state.sparse_index = sparse_index
    st.session_state.pipeline = SmartDocsPipeline(
        dense_store=dense_store,
        sparse_index=sparse_index,
        llm_client=llm_client
    )

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

# ==========================================
# SIDEBAR: DOCUMENT VAULT & INDEX CONTROLS
# ==========================================

with st.sidebar:
    st.title("📁 Document Vault")
    st.markdown("Upload files to index into **ChromaDB + BM25**.")

    uploaded_files = st.file_uploader(
        "Upload PDFs, TXT, DOCX, or MD",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            with st.spinner("Parsing, Chunking, and Indexing..."):
                total_new_chunks = 0
                for file in uploaded_files:
                    saved_path = UPLOAD_DIR / file.name
                    with open(saved_path, "wb") as f:
                        f.write(file.getbuffer())

                    # Parse & Chunk
                    docs = DocumentLoader.load_file(str(saved_path))
                    chunks = st.session_state.chunker.chunk_documents(docs)

                    # Index in both engines
                    st.session_state.dense_store.add_chunks(chunks)
                    st.session_state.sparse_index.add_chunks(chunks)
                    total_new_chunks += len(chunks)

                st.success(f"✅ Successfully indexed {total_new_chunks} chunks across {len(uploaded_files)} file(s)!")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.subheader("📊 Active Indexed Files")
    existing_files = list(UPLOAD_DIR.glob("*"))
    if existing_files:
        for f in existing_files:
            st.markdown(f"📄 **{f.name}** ({round(f.stat().st_size / 1024, 1)} KB)")
        
        if st.button("🗑️ Clear Entire Database", use_container_width=True):
            st.session_state.dense_store.clear()
            st.session_state.sparse_index.clear()
            for f in existing_files:
                try:
                    f.unlink()
                except Exception:
                    pass
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.warning("Database reset complete!")
            time.sleep(1)
            st.rerun()
    else:
        st.info("No documents uploaded yet.")

# ==========================================
# MAIN DASHBOARD: TABS
# ==========================================

st.title("🧠 SmartDocs — Advanced Multi-Strategy RAG")
st.caption("Production Q&A System with Hybrid Search (BM25 + Dense), Intent Routing, Cross-Encoder Reranking, and Citations.")

tab_chat, tab_inspector = st.tabs(["💬 Interactive Q&A Chat", "🔬 Under-the-Hood Pipeline Inspector"])

# ------------------------------------------
# TAB 1: INTERACTIVE CHAT
# ------------------------------------------
with tab_chat:
    # Display previous message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Source Citations"):
                    for c in msg["citations"]:
                        st.markdown(f"- 📄 **{c['source']}** (Page {c['page']}) — `ID: {c['chunk_id']}`")

    # Chat Input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Execute Pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching & Verifying Facts..."):
                # Extract history
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                
                # Run full SmartDocs pipeline
                result = st.session_state.pipeline.answer_query(
                    query=user_input,
                    chat_history=history,
                    top_k=3
                )

                answer_text = result["answer"]
                citations = result["citations"]
                st.session_state.last_trace = result["trace"]

                # Display answer
                st.markdown(answer_text)

                if citations:
                    with st.expander("📚 Source Citations"):
                        for c in citations:
                            st.markdown(f"- 📄 **{c['source']}** (Page {c['page']}) — `ID: {c['chunk_id']}`")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "citations": citations
        })

# ------------------------------------------
# TAB 2: PIPELINE OBSERVABILITY INSPECTOR
# ------------------------------------------
with tab_inspector:
    st.subheader("🔬 Real-Time Pipeline Observability")
    st.caption("Inspect the internal state, decisions, and rankings for the most recent query.")

    trace = st.session_state.last_trace

    if trace:
        col1, col2, col3 = st.columns(3)
        with col1:
            intent_val = trace.get("intent", "N/A")
            badge_color = "🟢" if intent_val == "document_query" else "🟡"
            st.metric("1. Classified Intent", f"{badge_color} {intent_val.upper()}")
        with col2:
            st.metric("2. Raw Candidates Retrieved", trace.get("retrieved_count", 0))
        with col3:
            st.metric("3. Quality Chunks After Rerank", trace.get("reranked_count", 0))

        st.markdown("---")
        
        st.markdown("#### 🔄 Query Transformations")
        st.markdown(f"- **Original User Query**: `{trace.get('original_query')}`")
        st.markdown(f"- **Rewritten Standalone Query**: `{trace.get('rewritten_query') or 'None (Fast-path chitchat)'}`")

        st.markdown("---")

        st.markdown("#### 🏆 Winning Chunks Passed to Generator")
        chunks_used = trace.get("chunks_used", [])
        if chunks_used:
            for i, c in enumerate(chunks_used, start=1):
                st.info(f"**Chunk #{i}** (`{c['chunk_id']}`) | **Score**: `{c['score']}` | **Source**: `{c['source']}` (Page {c['page']})")
        else:
            st.warning("No chunks passed the 0.35 relevance quality threshold.")

        st.markdown("---")
        st.markdown(f"**🛡️ Hallucination Guardrail Status**: {'✅ Entailment Verified' if trace.get('guardrail_verified') else '❌ Flagged as Hallucination'}")

    else:
        st.info("Ask a question in the chat tab to inspect the pipeline trace here!")
