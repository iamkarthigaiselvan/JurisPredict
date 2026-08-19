import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_MODEL = "openai/gpt-oss-20b"
    LLM_TEMPERATURE = 0.2

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 4

    APP_NAME = "JurisPredict"
    VERSION = "0.1.0"
    AUTHOR = "Karthi - LPU CSE (AI/ML)"


if not Config.GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. "
        "Check your .env file."
    )