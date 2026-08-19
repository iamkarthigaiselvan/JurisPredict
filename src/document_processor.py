from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import streamlit as st


class DocumentProcessor:
    """Handles PDF extraction and text chunking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

    def extract_text(self, pdf_file) -> str:
        """Extract text from a PDF."""

        try:
            pdf_reader = PdfReader(pdf_file)

            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            return text.strip()

        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def create_chunks(self, text: str) -> List[str]:
        """Split extracted text into chunks."""

        if not text:
            return []

        return self.text_splitter.split_text(text)

    def process(self, pdf_file) -> Tuple[str, List[str]]:
        """Complete PDF processing pipeline."""

        text = self.extract_text(pdf_file)

        if len(text) < 200:
            raise ValueError(
                "PDF appears to be image-based or contains no "
                "extractable text. Please use a text-based PDF."
            )

        chunks = self.create_chunks(text)

        return text, chunks