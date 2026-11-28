import sqlite3
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

from app.core.config import get_settings
from app.rag.vectorstore import get_vector_store
from app.prompts.loaders import get_prompt
from app.llm.models import get_chat_model
from app.tools.knowledge.handbook import file_search

from langgraph.checkpoint.sqlite import SqliteSaver
from app.middleware.stack import default_stack

# SQLite conversation memory
conn = sqlite3.connect(
    database="memory.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


class ClauseAgent:

    def __init__(self) -> None:

        self.settings = get_settings()
        self.store = get_vector_store(settings=self.settings)

        # Default conversation ID
        self.thread_id = str(uuid4())

        # @tool(response_format="content_and_artifact")
        # def search_handbook(query: str):
        #     """Search the shop handbook for information about
        #     refunds, payments, delivery, customer policies,
        #     and other shop-related questions.

        #     Args:
        #         query: The customer's question.

        #     Returns:
        #         Relevant handbook text and source information.
        #     """

        #     results = self.store.similarity_search_with_score(
        #         query=query,
        #         k=self.settings.top_k
        #     )

        #     if not results:
        #         return "Nothing Found", []

        #     texts = []
        #     sources = []

        #     for doc, score in results:

        #         texts.append(doc.page_content)

        #         sources.append(
        #             {
        #                 "page": doc.metadata.get("page"),
        #                 "score": round(float(score), 4),
        #             }
        #         )

        #     text = "\n\n".join(texts)

        #     return text, sources

        # Create agent
        self.agent = create_agent(
            model=get_chat_model(settings=self.settings),
            tools=[
                file_search(store=self.store, top_k=self.settings.top_k)
                ],
            system_prompt=get_prompt(
                name="support/system",
                version=self.settings.system_prompt_version
            ),
            middleware=default_stack(settings=self.settings),
            checkpointer=checkpointer,
        )

    # Ask question
    def ask(
        self,
        question: str,
        thread_id: str | None = None
    ) -> Any:

        # Use supplied conversation ID or default ID
        if thread_id is None:
            thread_id = self.thread_id

        result = self.agent.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            },
        )

        response = result["messages"][-1].content

        if isinstance(response, list):
            response = " ".join(
                block["text"]
                for block in response
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return response

    # Reset conversation memory
    def reset_memory(self) -> None:

        self.thread_id = str(uuid4())

    # Close resources
    def close(self) -> None:

        self.store.client.close()
        conn.close()