
import chromadb

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Any


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore = None
        self.client = None

    def create_index(self, texts: List[str]) -> None:
        """Create an in-memory ChromaDB vector index."""

        if not texts:
            raise ValueError("No text chunks provided for indexing.")

        # Prepare metadata
        metadatas = [
            {
                "source": f"chunk_{i}",
                "index": i
            }
            for i in range(len(texts))
        ]

        # Create a fresh in-memory ChromaDB client
        self.client = chromadb.EphemeralClient(
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create vector store
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name="legal_documents",
            client=self.client
        )

    def get_retriever(self, k: int = 4):
        """Return a retriever for similarity search."""

        if self.vectorstore is None:
            raise ValueError(
                "Vector store not initialized. "
                "Call create_index first."
            )

        return self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Any]:
        """Perform direct similarity search for debugging."""

        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        return self.vectorstore.similarity_search(
            query,
            k=k
        )
