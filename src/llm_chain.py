from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import Config


class LegalRAGChain:
    """RAG chain using Groq LLM and ChromaDB retrieval."""

    def __init__(self, retriever):
        self.retriever = retriever

        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
        )

        self.prompt = PromptTemplate(
            template="""You are JurisPredict, an AI legal research assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, say:
"I cannot find sufficient information in the provided document."

Do not invent legal facts, cases, citations, or conclusions.

Context:
{context}

Question:
{question}

Answer:""",
            input_variables=["context", "question"],
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self.prompt
            }
        )

    def ask(self, question: str):
        """Ask a question and return answer + source documents."""

        result = self.qa_chain.invoke({
            "query": question
        })

        return {
            "answer": result["result"],
            "source_documents": result.get(
                "source_documents", []
            )
        }