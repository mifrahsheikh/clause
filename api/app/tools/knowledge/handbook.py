from langchain.tools import tool
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from app.schemas.retrieval import Source

def file_search(store: QdrantVectorStore, top_k: int):

    @tool(response_format="content_and_artifact")
    def search_handbook(query: str):
        """Search the shop handbook for information about
        refunds, payments, delivery, customer policies,
        and other shop-related questions.

        Args:
            query: The customer's question.

        Returns:
            Relevant handbook text and source information.
        """

        results = store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        if not results:
            return "Nothing Found", []

        texts = []
        sources = []

        for doc, score in results:

            texts.append(doc.page_content)

            sources.append({
                "page": doc.metadata.get("page"),
                "score": round(float(score), 4),
            })

        text = "\n\n".join(texts)
        sources = [
            Source(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                score=round(score, 4)
            )

            for doc, score in results
        ]

        return text, sources

    # IMPORTANT: Return the decorated tool
    return search_handbook