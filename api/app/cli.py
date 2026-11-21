from uuid import uuid4

from app.services.agent import ClauseAgent


def main() -> None:

    clause_agent = ClauseAgent()

    try:

        

        choice = input(
            "Enter 1 for a new conversation "
            "or 2 to continue an existing conversation: "
        ).strip()

        if choice not in {"1", "2"}:
            print("Invalid choice. Please enter 1 or 2.")
            return

        if choice == "2":

            thread_id = input(
                "Enter conversation ID: "
            ).strip()

            if not thread_id:
                print("Invalid conversation ID.")
                return

        else:

            thread_id = str(uuid4())

            print("\nNew conversation created.")

        print(f"Conversation ID: {thread_id}")
        print("\nType 'exit' to end the conversation.\n")

        while True:

            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in {
                "quit",
                "bye",
                "exit",
            }:
                print("Clause: Goodbye!")
                break

            result = clause_agent.ask(
                question=query,
                thread_id=thread_id,
            )

            print(f"\nAI: {result}")

            # Display sources when available
        

    except (KeyboardInterrupt, EOFError):

        print("\nConversation ended.")

    finally:

        clause_agent.close()


if __name__ == "__main__":
    main()