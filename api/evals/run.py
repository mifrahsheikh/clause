import json
import time
from datetime import datetime
from pathlib import Path

from app.services.agent import ClauseAgent

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVALS_DIR / "golden.jsonl"
RESULTS_DIR = EVALS_DIR / "results"


REFUSALS = [
    "don't know",
    "do not know",
    "not mention",
    "doesn't mention",
    "doesn't cover",
    "does not cover",
    "no information",
]


CONFIG_KEYS = [
    "chat_provider",
    "chat_model",
    "embed_provider",
    "embed_model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "score_threshold",
    "system_prompt_version",
]


def clean(text):
    return str(text).lower().replace(",", "").replace("’", "'")


def has(text, word):
    return any(option in text for option in word.split("|"))


def rate(results):
    return round(sum(results) / len(results), 3) if results else None


def pct(value):
    return f"{value:.0%}" if value is not None else "n/a"


def load_tests():

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_FILE}")

    with open(GOLDEN_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_test(agent, t):

    start = time.perf_counter()

    question = t["question"]

    # Give every evaluation question a separate conversation.
    thread_id = f"eval-{t['id']}"

    result = agent.ask(
        question=question,
        thread_id=thread_id,
    )

    # Current ClauseAgent.ask() returns a string.
    answer = str(result).strip()

    record = {
        "id": t["id"],
        "type": t["type"],
        "question": question,
        "answer": answer,
        "sources": [],
        "seconds": round(time.perf_counter() - start, 2),
    }

    answer_clean = clean(answer)

    # -----------------------------------------
    # Answerable questions
    # -----------------------------------------

    if t["type"] == "answerable":
        # Retrieve chunks for retrieval evaluation.
        # This is a separate diagnostic retrieval.
        chunks = agent.store.similarity_search(
            question,
            k=agent.settings.top_k,
        )

        context = clean(" ".join(chunk.page_content for chunk in chunks))

        retrieved_pages = [chunk.metadata.get("page") for chunk in chunks]

        record["sources"] = [
            {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page"),
            }
            for chunk in chunks
        ]

        record["page_hit"] = any(
            page in t["expected_pages"] for page in retrieved_pages
        )

        record["fact_in_chunks"] = all(has(context, word) for word in t["must_include"])

        record["passed"] = all(has(answer_clean, word) for word in t["must_include"])

        record["blame"] = (
            "generator: the fact WAS in the chunks"
            if record["fact_in_chunks"]
            else "retriever: the fact was NOT in the chunks"
        )

    # -----------------------------------------
    # Out-of-scope questions
    # -----------------------------------------

    elif t["type"] == "out_of_scope":
        record["passed"] = NO_ANSWER_IN(answer_clean)

        record["blame"] = "should have returned exactly I don't know."

    # -----------------------------------------
    # Restricted information / leakage tests
    # -----------------------------------------

    else:
        record["passed"] = not any(
            has(answer_clean, word) for word in t["must_not_include"]
        )

        record["blame"] = "leaked staff-only information"

    if record["passed"]:
        record.pop("blame", None)

    return record


def NO_ANSWER_IN(answer):

    return answer.strip() == "i don't know."


def summarize(records):

    answerable = [r for r in records if r["type"] == "answerable"]

    refusal = [r for r in records if r["type"] == "out_of_scope"]

    other = [r for r in records if r["type"] not in ("answerable", "out_of_scope")]

    return {
        "total": len(records),
        "passed": sum(r["passed"] for r in records),
        "retrieval_hit_rate": rate([r["page_hit"] for r in answerable]),
        "fact_in_chunks_rate": rate([r["fact_in_chunks"] for r in answerable]),
        "answer_correctness": rate([r["passed"] for r in answerable]),
        "refusal_rate": rate([r["passed"] for r in refusal]),
        "leak_rate": rate([not r["passed"] for r in other]),
    }


def save_report(report, started):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = RESULTS_DIR / (f"{started:%Y-%m-%d_%H-%M-%S}.json")

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return path


def main():

    started = datetime.now()

    tests = load_tests()

    agent = ClauseAgent()

    records = []

    try:
        for t in tests:
            record = run_test(agent, t)

            records.append(record)

            print(
                "✓" if record["passed"] else "✗",
                t["id"],
            )

            if not record["passed"]:
                print("   answer:", record["answer"])

                print("   blame:", record.get("blame", "unknown"))

    finally:
        agent.close()

    summary = summarize(records)

    print(f"\nPassed: {summary['passed']}/{summary['total']}")

    print("Retrieval hit rate (page):", pct(summary["retrieval_hit_rate"]))

    print("Fact in chunks:", pct(summary["fact_in_chunks_rate"]))

    print("Answer correctness:", pct(summary["answer_correctness"]))

    print("Refusal rate:", pct(summary["refusal_rate"]))

    print("Leak rate:", pct(summary["leak_rate"]))

    settings = agent.settings.model_dump()

    report = {
        "started_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round(
            (datetime.now() - started).total_seconds(),
            1,
        ),
        "config": {key: settings.get(key) for key in CONFIG_KEYS},
        "summary": summary,
        "results": records,
    }

    report_path = save_report(report, started)

    print("\nSaved:", report_path.relative_to(EVALS_DIR.parent))


if __name__ == "__main__":
    main()
