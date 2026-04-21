import json
import streamlit as st

from src.data_loader import load_all_documents
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import TextEmbedder
from src.llm_generator import LLMGenerator
from src.pipeline import run_rag_pipeline


st.set_page_config(
    page_title="ACity RAG Chatbot",
    page_icon="assets/logo.svg",
    layout="wide"
)


def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #08111f 0%, #0b1220 100%);
        color: #f3f4f6;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    h1, h2, h3, h4 {
        color: #f8fafc;
        letter-spacing: -0.4px;
    }

    p, label, div {
        color: #d1d5db;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(16,24,40,0.96), rgba(15,23,42,0.90));
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin-bottom: 0.35rem;
        line-height: 1.55;
    }

    .mini-card {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }

    .mini-label {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }

    .mini-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .section-space {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }

    .answer-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16), rgba(34, 197, 94, 0.12));
        border: 1px solid rgba(74, 222, 128, 0.28);
        border-radius: 18px;
        padding: 1.15rem 1.2rem;
        margin-top: 0.35rem;
        margin-bottom: 0.5rem;
    }

    .answer-text {
        font-size: 1.16rem;
        font-weight: 600;
        color: #dcfce7;
        line-height: 1.55;
    }

    .source-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.24);
        color: #bae6fd;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.3rem;
        margin-bottom: 0.6rem;
    }

    div[data-baseweb="input"] > div {
        background-color: rgba(30, 41, 59, 0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 14px !important;
    }

    div[data-baseweb="input"] input {
        color: #f8fafc !important;
        font-size: 1.02rem !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.56);
    }

    div[data-testid="stExpander"] details summary {
        font-weight: 600;
        color: #e2e8f0;
    }

    div[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.97);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.76);
        border-radius: 10px 10px 0 0;
        padding: 0.48rem 0.88rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.14) !important;
        color: #e0f2fe !important;
    }

    textarea {
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


inject_custom_css()


@st.cache_resource
def load_models():
    embedder = TextEmbedder()
    llm = LLMGenerator(model_name="HuggingFaceTB/SmolLM2-135M-Instruct")
    return embedder, llm


@st.cache_data
def prepare_data():
    docs = load_all_documents()
    cleaned_docs = clean_documents(docs)
    chunks = chunk_documents(
        cleaned_docs,
        strategy="fixed",
        chunk_size=500,
        overlap=100
    )
    texts = [chunk["text"] for chunk in chunks]
    return docs, cleaned_docs, chunks, texts


@st.cache_resource
def build_embeddings(texts):
    embedder, _ = load_models()
    chunk_embeddings = embedder.embed_texts(texts)
    return chunk_embeddings


def display_chunk(chunk, index):
    st.markdown(f"#### Chunk {index}")
    st.write(f"**Source Type:** {chunk.get('source_type', 'N/A')}")
    st.write(f"**Source Name:** {chunk.get('source_name', 'N/A')}")
    st.write(f"**Chunk ID:** {chunk.get('chunk_id', 'N/A')}")
    st.write(f"**Cosine Similarity:** {chunk.get('cosine_similarity', 0):.4f}")
    st.write(f"**Keyword Overlap:** {chunk.get('keyword_overlap', 0):.4f}")
    st.write(f"**Final Score:** {chunk.get('final_score', 0):.4f}")

    if chunk.get("page_number") is not None:
        st.write(f"**Page Number:** {chunk['page_number']}")
    if chunk.get("row_number") is not None:
        st.write(f"**Row Number:** {chunk['row_number']}")

    st.text_area(
        f"Chunk Text {index}",
        chunk["text"],
        height=180,
        key=f"chunk_text_{index}"
    )


def main():
    with st.spinner("Loading documents, model, and embeddings..."):
        docs, cleaned_docs, chunks, texts = prepare_data()
        embedder, llm = load_models()
        chunk_embeddings = build_embeddings(texts)

    st.sidebar.header("System Settings")
    top_k = st.sidebar.slider("Top-K Retrieval", min_value=1, max_value=10, value=5)
    show_debug = st.sidebar.checkbox("Show Debug Sections", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")
    st.sidebar.write("**Name:** Chizota Diamond Chizzy")
    st.sidebar.write("**Index:** 10022200128")
    st.sidebar.write("**Mode:** RAG")
    st.sidebar.write("**LLM:** SmolLM2-135M-Instruct")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Summary")
    st.sidebar.write(f"Raw documents: {len(docs)}")
    st.sidebar.write(f"Cleaned documents: {len(cleaned_docs)}")
    st.sidebar.write(f"Chunks: {len(chunks)}")

    col1, col2 = st.columns([1, 8])

    with col1:
        st.image("assets/logo.svg", width=92)

    with col2:
        st.markdown("""
        <div class="hero-card">
            <div class="hero-title">ACity RAG Chatbot</div>
            <p class="hero-sub">
                A Retrieval-Augmented Generation system for Ghana Election and 2025 Budget question answering.
            </p>
            <p class="hero-sub">
                <strong>Student:</strong> Chizota Diamond Chizzy |
                <strong>Index:</strong> 10022200128
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">Raw documents</div>
            <div class="mini-value">{len(docs)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">Cleaned documents</div>
            <div class="mini-value">{len(cleaned_docs)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">Total chunks</div>
            <div class="mini-value">{len(chunks)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="mini-card">
            <div class="mini-label">Retrieval top-k</div>
            <div class="mini-value">{top_k}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="small-note">Ask a question about the Ghana Election or the 2025 Budget.</p>',
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Question input",
        label_visibility="collapsed",
        placeholder="Example: Who won the 2020 Ghana presidential election?"
    )

    if question:
        with st.spinner("Running RAG pipeline..."):
            result = run_rag_pipeline(
                query=question,
                embedder=embedder,
                chunk_embeddings=chunk_embeddings,
                chunk_docs=chunks,
                llm=llm,
                top_k=top_k
            )

        st.subheader("Final Answer")

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-text">{result["final_answer"]}</div>
        </div>
        <div class="source-pill">Answer Source: {result["answer_source"]}</div>
        """, unsafe_allow_html=True)

        if show_debug:
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["User Query", "Retrieved Chunks", "Selected Context", "Final Prompt", "Export"]
            )

            with tab1:
                st.subheader("User Query")
                st.write(result["query"])

            with tab2:
                st.subheader("Retrieved Chunks")
                for i, chunk in enumerate(result["retrieved_chunks"], start=1):
                    with st.expander(f"Retrieved Chunk {i}", expanded=(i == 1)):
                        display_chunk(chunk, i)

            with tab3:
                st.subheader("Selected Context")
                st.text_area(
                    "Context Passed to the Answering Stage",
                    result["selected_context"],
                    height=300,
                    key="selected_context_box"
                )

            with tab4:
                st.subheader("Final Prompt")
                st.text_area(
                    "Prompt Used for Final Answer",
                    result["final_prompt"],
                    height=320,
                    key="final_prompt_box"
                )

            with tab5:
                st.subheader("Export Result")
                export_payload = {
                    "query": result["query"],
                    "answer": result["final_answer"],
                    "answer_source": result["answer_source"],
                    "selected_context": result["selected_context"],
                    "final_prompt": result["final_prompt"],
                    "retrieved_chunks": result["retrieved_chunks"]
                }

                st.download_button(
                    label="Download Result as JSON",
                    data=json.dumps(export_payload, indent=2),
                    file_name="rag_result.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()