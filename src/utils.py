import streamlit as st
from typing import List, Dict, Any


def validate_pdf(uploaded_file) -> bool:
    """Validate uploaded PDF."""

    if uploaded_file is None:
        return False

    if uploaded_file.type != "application/pdf":
        st.error("Please upload a valid PDF file")
        return False

    return True


def format_source_document(
    doc,
    index: int
) -> Dict[str, Any]:
    """Format a source document for display."""

    return {
        "index": index + 1,
        "content": (
            doc.page_content[:400] + "..."
            if len(doc.page_content) > 400
            else doc.page_content
        ),
        "metadata": doc.metadata
    }


def calculate_metrics(
    text: str,
    chunks: List[str]
) -> Dict[str, int]:
    """Calculate document metrics."""

    return {
        "total_chars": len(text),
        "total_chunks": len(chunks),
        "avg_chunk_size": (
            sum(len(c) for c in chunks)
            // max(len(chunks), 1)
        )
    }