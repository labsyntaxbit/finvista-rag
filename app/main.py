"""FinVista Capital - Enterprise Financial Intelligence Assistant (Streamlit UI)."""

import streamlit as st
from pathlib import Path

from app.rag.document_loader import DocumentLoader
from app.rag.chunker import TextChunker
from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever
from app.rag.generator import ResponseGenerator
from app.rag.memory import ConversationMemory
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="FinVista Intelligence Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a365d; }
    .sub-header { color: #4a5568; margin-bottom: 1.5rem; }
    .citation-box {
        background: #f7fafc; border-left: 4px solid #3182ce;
        padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_vector_store():
    return VectorStore()


@st.cache_resource
def get_retriever():
    return Retriever(get_vector_store())


def init_session_state():
    defaults = {
        "memory": ConversationMemory(),
        "messages": [],
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def process_uploaded_file(uploaded_file) -> int:
    loader = DocumentLoader()
    chunker = TextChunker()
    vector_store = get_vector_store()

    content = uploaded_file.read()
    document = loader.load_from_bytes(uploaded_file.name, content)
    chunks = chunker.chunk_document(document)
    count = vector_store.add_chunks(chunks)

    upload_path = Path(settings.upload_dir) / uploaded_file.name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(content)

    return count


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/bank-building.png", width=48)
        st.title("FinVista Capital")
        st.caption("Enterprise Financial Intelligence")

        st.divider()
        st.subheader("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload financial documents (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload annual reports, analyst reports, compliance manuals, etc.",
        )

        if uploaded_files and st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing and indexing documents..."):
                total_chunks = 0
                for f in uploaded_files:
                    try:
                        chunks = process_uploaded_file(f)
                        total_chunks += chunks
                        st.success(f"✅ {f.name} — {chunks} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                        logger.error("Failed to process %s: %s", f.name, e)
                if total_chunks:
                    st.info(f"Total: {total_chunks} chunks indexed")

        st.divider()
        st.subheader("📚 Indexed Documents")
        vector_store = get_vector_store()
        docs = vector_store.list_documents()
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc}", help=f"Remove {doc}"):
                        vector_store.delete_document(doc)
                        st.rerun()
        else:
            st.info("No documents indexed yet.")

        st.divider()
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric("Documents", len(docs))
        with stats_col2:
            st.metric("Chunks", vector_store.get_document_count())

        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(f"LLM: {settings.llm_provider.upper()}")
        st.caption(f"Embeddings: {settings.embedding_model}")


def render_chat():
    st.markdown('<p class="main-header">📊 FinVista Intelligence Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your enterprise financial documents</p>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Source Citations"):
                    for i, cite in enumerate(msg["citations"], 1):
                        st.markdown(
                            f'<div class="citation-box">'
                            f'<strong>Source {i}:</strong> {cite["filename"]} '
                            f'(Chunk {cite["chunk_index"] + 1}, Relevance: {cite["score"]})<br>'
                            f'<em>{cite["excerpt"]}</em></div>',
                            unsafe_allow_html=True,
                        )

    if prompt := st.chat_input("Ask a question about your financial documents..."):
        vector_store = get_vector_store()
        if vector_store.get_document_count() == 0:
            st.warning("Please upload and process documents before asking questions.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.memory.add_user_message(prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    retriever = get_retriever()
                    chunks = retriever.retrieve(prompt)
                    context = retriever.format_context(chunks)
                    citations = retriever.get_citations(chunks)

                    generator = ResponseGenerator()
                    history = st.session_state.memory.get_messages_for_llm()[:-1]
                    response = generator.generate(prompt, context, history)

                    st.markdown(response)

                    if citations:
                        with st.expander("📎 Source Citations", expanded=True):
                            for i, cite in enumerate(citations, 1):
                                st.container(border=True)
                    
                                st.markdown(f"### 📄 Source {i}")
                                st.write(f"**File:** {cite['filename']}")
                                st.write(f"**Chunk:** {cite['chunk_index'] + 1}")
                                st.write(f"**Relevance:** {cite['score']:.4f}")
                                st.write("**Excerpt:**")
                                st.info(cite["excerpt"])
                    
                                st.divider()

                    st.session_state.memory.add_assistant_message(response, citations)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response, "citations": citations}
                    )
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    logger.error("Generation failed: %s", e)


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()