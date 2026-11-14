# Indexing

from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from core.config import get_settings, get_qdrant_client, get_embeddings


class Source(BaseModel):
    source: str
    page: int | None = None
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()

        self.client = get_qdrant_client(settings=self.settings)

        if not self.client.collection_exists(
            collection_name=self.settings.collection_name
        ):
            raise RuntimeError(
                f"Collection {self.settings.collection_name} not found."
            )

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding=get_embeddings(settings=self.settings),
        )

        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.chat_temperature,
        )

    def ask(self, question: str) -> Answer:
        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty.")

        # 1. Retrieve relevant documents from Qdrant
        results = self.store.similarity_search_with_score(
            query=question,
            k=self.settings.top_k,
        )

        if not results:
            return Answer(answer="I don't know.", sources=[])

        # 2. Prepare context
        context = "\n\n".join(
            doc.page_content for doc, score in results
        )

        # 3. Prepare messages
        messages = [
            {
                "role": "system",
                "content": self.settings.system_prompt,
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        # 4. Generate answer
        response = self.llm.invoke(messages)

        # 5. Return answer and sources
        return Answer(
            answer=response.content,
            sources=[
                Source(
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page"),
                    score=round(float(score), 4),
                )
                for doc, score in results
            ],
        )

    def close(self) -> None:
        self.client.close()