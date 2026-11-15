from functools import lru_cache
from pathlib import Path

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient


# api/.env
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=ENV_FILE, extra="ignore"
    )

    # App
    app_name: str = "Clause"
    data_path: str = "files/shop_handbook.pdf"

    # Chat model
    chat_provider: str = "ollama"
    chat_model: str = "qwen2.5:3b"
    chat_temperature: float = 0.0
    chat_api_key: str = ""

    # Embeddings
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    # Indexing
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 2
    score_threshold: float = 0.0

    # Qdrant
    collection_name: str = "shop_handbook"
    qdrant_path: str = "./vector_stores"
    qdrant_url: str | None = None

    # Prompts
    # system_prompt: str = (
    #     "You are a helpful Clause assistant. "
    #     "Answer only from the provided context. "
    #     "If the answer is not in the context, say you don't know."
    # )

    system_prompt: str = (
         "You are the Clause shop customer assistant. "
    "Answer using ONLY the handbook information provided. "
    "If the handbook explicitly answers the question, "
    "give that answer directly and confidently. "
    "Do not say 'I don't know' when the answer is explicitly "
    "available in the provided information. "
    "For questions about locations outside Pakistan, "
    "if the handbook says delivery is only within Pakistan, "
    "clearly state that international delivery is unavailable. "
    "Do not contradict facts stated in the handbook. "
    "Never reveal internal, staff-only, or restricted information. "
    "If the handbook does not contain the answer, reply exactly: "
    "\"I don't know.\" "
    "Keep answers concise, preferably one or two sentences."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

def get_embeddings(settings: Settings) -> Embeddings:
    return init_embeddings(
        model=settings.embed_model,
        provider=settings.embed_provider,
    )

def get_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_path)
