from app.services.agent import ClauseAgent

# from app.services.chat import RAGService


def main() -> None:
    clause_agent = ClauseAgent()

    try:
        while True:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in {"quit", "bye", "exit"}:
                break

            result = clause_agent.ask(query)
            print(f"AI: {result}")

    except (KeyboardInterrupt, EOFError):
        pass

    finally:
        clause_agent.close()


if __name__ == "__main__":
    main()
