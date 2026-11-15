from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from app.core.config import get_settings, get_qdrant_client, get_embeddings


NO_ANSWER = "I don't know."

class Source(BaseModel):
    source: str
    page: int | None = None
    score: float

class Answer(BaseModel):
    answer: str
    sources: list[Source]

class RAGService():
    def __init__(self) -> None:
        self.settings = get_settings()

        self.client = get_qdrant_client(self.settings)
        if not self.client.collection_exists(self.settings.collection_name):
            raise RuntimeError(f"Collection {self.settings.collection_name} not found!")

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=get_embeddings( self.settings)
        )

        # LLM
        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            api_key=self.settings.chat_api_key,
            # temperature=self.settings.chat_temperature,
        )

    def ask(self, question: str):
        question = question.strip()
        if not question:
            raise ValueError("Question must be not empty")

        results = self.store.similarity_search_with_score(
            query=question,
            k=self.settings.top_k,
            score_threshold=self.settings.score_threshold
        )

        if not results:
            return Answer(answer=NO_ANSWER, sources=[])

        context = "\n\n".join(doc.page_content for doc, _ in results)

        messages = [
            {"role": "system", "content": self.settings.system_prompt},
            # {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            {"role": "user", "content": f"Handbook text:\n{context}\n\nCustomer question: {question}"},
        ]

        return Answer(
            answer=self.llm.invoke(messages).text,
            sources=[
                Source(
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    score=round(score, 4)
                )
                for doc, score in results
            ]
        )

    def close(self):
        self.client.close()
