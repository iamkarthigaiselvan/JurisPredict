import streamlit as st

from config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_chain import LegalRAGChain
from src.utils import validate_pdf, format_source_document, calculate_metrics



st.set_page_config(
    page_title="JurisPredict",
    page_icon="⚖️",
    layout="wide"
)




if "document_processor" not in st.session_state:
    st.session_state.document_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []




st.title("⚖️ JurisPredict")
st.caption(
    "AI-powered legal research assistant using Retrieval-Augmented Generation"
)

st.divider()




with st.sidebar:

    st.header("📄 Legal Document")

    uploaded_file = st.file_uploader(
        "Upload a legal PDF",
        type=["pdf"]
    )

    st.divider()

    st.subheader("Configuration")

    st.write(
        f"**Embedding:** `{Config.EMBEDDING_MODEL}`"
    )

    st.write(
        f"**LLM:** `{Config.LLM_MODEL}`"
    )

    st.write(
        f"**Chunk size:** `{Config.CHUNK_SIZE}`"
    )

    st.write(
        f"**Chunk overlap:** `{Config.CHUNK_OVERLAP}`"
    )

    st.write(
        f"**Top-K:** `{Config.TOP_K_RETRIEVAL}`"
    )




if uploaded_file is not None:

    if validate_pdf(uploaded_file):

        if st.session_state.get("processed_file") != uploaded_file.name:

            with st.spinner("Processing legal document..."):

                try:

                    text, chunks = (
                        st.session_state.document_processor.process(
                            uploaded_file
                        )
                    )

                    st.session_state.document_text = text
                    st.session_state.chunks = chunks

                    # Create vector store
                    with st.spinner(
                        "Creating embeddings and vector database..."
                    ):

                        vector_store = VectorStoreManager(
                            model_name=Config.EMBEDDING_MODEL
                        )

                        vector_store.create_index(chunks)

                        retriever = vector_store.get_retriever(
                            k=Config.TOP_K_RETRIEVAL
                        )

                        rag_chain = LegalRAGChain(
                            retriever
                        )

                        st.session_state.vector_store = vector_store
                        st.session_state.rag_chain = rag_chain
                        st.session_state.processed_file = (
                            uploaded_file.name
                        )

                    st.success("Document processed successfully!")

                except Exception as e:

                    st.error(
                        f"Error processing document: {str(e)}"
                    )




if st.session_state.document_text:

    metrics = calculate_metrics(
        st.session_state.document_text,
        st.session_state.chunks
    )

    st.subheader("📊 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Characters",
            f"{metrics['total_chars']:,}"
        )

    with col2:
        st.metric(
            "Chunks",
            f"{metrics['total_chunks']:,}"
        )

    with col3:
        st.metric(
            "Avg. Chunk Size",
            f"{metrics['avg_chunk_size']:,}"
        )

    st.divider()




st.subheader("🔎 Ask about the document")

if st.session_state.rag_chain is None:

    st.info(
        "Upload a legal PDF first, then ask a question about it."
    )

else:

    question = st.text_input(
        "Your question",
        placeholder=(
            "What are the main grounds for appeal?"
        )
    )

    if st.button("Ask JurisPredict", type="primary"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching document and generating answer..."):

                try:

                    result = st.session_state.rag_chain.ask(
                        question
                    )

                    # Answer
                    st.subheader("💡 Answer")

                    st.write(result["answer"])

                    # Sources
                    st.subheader("📚 Sources")

                    sources = result.get(
                        "source_documents",
                        []
                    )

                    if not sources:

                        st.info(
                            "No source documents were returned."
                        )

                    else:

                        for i, doc in enumerate(
                            sources
                        ):

                            source = format_source_document(
                                doc,
                                i
                            )

                            with st.expander(
                                f"Source {source['index']}"
                            ):

                                st.write(
                                    source["content"]
                                )

                                if source["metadata"]:

                                    st.caption(
                                        f"Metadata: "
                                        f"{source['metadata']}"
                                    )

                except Exception as e:

                    st.error(
                        f"Error generating answer: {str(e)}"
                    )



st.divider()

st.caption(
    "JurisPredict v0.1.0 — AI legal research assistant. "
    "This tool is for research assistance and does not "
    "constitute legal advice."
)