from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from pydantic import BaseModel
from app.prompts.loaders import get_prompt

from app.core.config import Settings, get_qdrant_client, get_embeddings


def get_vector_store(settings: Settings) -> QdrantVectorStore:
    client = get_qdrant_client(settings)
    if not client.collection_exists(settings.collection_name):
        raise RuntimeError(f"Collection {settings.collection_name} not found!")
    
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=get_embeddings(settings),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )