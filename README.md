# ⚖️ JurisPredict

https://jurispredict-beqjmyc2a2gzeppejdexue.streamlit.app/

## AI-Powered Legal Research Assistant

JurisPredict is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about legal judgments and receive answers grounded in the source document.

##  How It Works

PDF
↓
Text Extraction
↓
Chunking
↓
MiniLM Embeddings
↓
ChromaDB
↓
Semantic Retrieval
↓
GPT-OSS 20B via Groq
↓
Answer + Sources

##  Tech Stack

- Python
- Streamlit
- Sentence Transformers
- MiniLM
- ChromaDB
- Groq
- GPT-OSS 20B
- PyPDF

##  Features

- Upload legal PDF documents
- Extract text from judgments
- Create semantic embeddings
- Search relevant legal passages
- Generate answers using an LLM
- Display retrieved source passages
- Reduce hallucination through document-grounded generation

##  Example

### Question

What constitutional rights were involved in the case?

### System

The question is converted into an embedding and compared against the legal document stored in ChromaDB.

The most relevant passages are then provided to GPT-OSS 20B to generate a grounded answer.

##  Disclaimer

JurisPredict is an educational and research project. It is not a substitute for professional legal advice.

## Author

Karthigaiselvan M