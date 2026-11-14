from services.chat import RAGService


def main():
    rag_services = RAGService()

    try:
        while True:
            query = input("You: ")

            if query.lower() in ["exit", "quit"]:
                break

            result = rag_services.ask(question=query)

            print("\nClause:", result.answer)

            print("\nSources:")
            for source in result.sources:
                print(
                    f"File: {source.source} | "
                    f"Page: {source.page} | "
                    f"Score: {source.score}"
                )

    finally:
        rag_services.close()


if __name__ == "__main__":
    main()