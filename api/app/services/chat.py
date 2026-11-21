# from langchain.chat_models import init_chat_model
# from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
# from pydantic import BaseModel
# from app.prompts.loaders import get_prompt
# from app.core.config import get_settings, get_qdrant_client, get_embeddings

# NO_ANSWER = "I don't know."

# class Source(BaseModel):
#     source: str
#     page: int | None = None
#     score: float

# class Answer(BaseModel):
#     answer: str
#     sources: list[Source]

# class RAGService():
#     def __init__(self) -> None:
#         self.settings = get_settings()
#         self.system_prompt = get_prompt(
#     name="support/system",
#     version=self.settings.system_prompt_version
# )

#         self.client = get_qdrant_client(self.settings)
#         if not self.client.collection_exists(self.settings.collection_name):
#             raise RuntimeError(f"Collection {self.settings.collection_name} not found!")

#         self.store = QdrantVectorStore(
#             client=self.client,
#             collection_name=self.settings.collection_name,
#             embedding=get_embeddings( self.settings),
#             sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
#             retrieval_mode=RetrievalMode.HYBRID,
#             vector_name="dense",
#             sparse_vector_name="sparse"
#         )

#         # LLM


#     def ask(self, question: str):
#         question = question.strip()
#         if not question:
#             raise ValueError("Question must be not empty")

#         results = self.store.similarity_search_with_score(
#             query=question,
#             k=self.settings.top_k,
#             # score_threshold=self.settings.score_threshold
#         )

#         if not results:
#             return Answer(answer="I don't know.", sources=[])

#         context = "\n\n".join(doc.page_content for doc, _ in results)

#         messages = [
#             {"role": "system", "content": self.system_prompt},
#             # {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
#             {"role": "user", "content": f"Handbook text:\n{context}\n\nCustomer question: {question}"},
#         ]

#         return Answer(
#             answer=self.llm.invoke(messages).text,
#             sources=[
#                 Source(
#                     source=doc.metadata.get("source", "unknown"),
#                     page=doc.metadata.get("page"),
#                     score=round(score, 4)
#                 )
#                 for doc, score in results
#             ]
#         )

#     def close(self):
#         self.client.close()