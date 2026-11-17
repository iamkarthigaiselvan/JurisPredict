
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
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore = None

        # Create one client and reuse it
        self.client = chromadb.EphemeralClient()

    def create_index(self, texts: List[str]) -> None:
        """Create or replace the vector index."""

        if not texts:
            raise ValueError("No text chunks provided.")

        metadatas = [
            {
                "source": f"chunk_{i}",
                "index": i
            }
            for i in range(len(texts))
        ]

        # Remove the previous collection if it exists
        try:
            self.client.delete_collection("legal_documents")
        except Exception:
            pass

        # Create the vector store using the existing client
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name="legal_documents",
            client=self.client
        )

    def get_retriever(self, k: int = 4):
        """Return a retriever."""

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
        """Perform similarity search."""

        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        return self.vectorstore.similarity_search(
            query,
            k=k
        )
