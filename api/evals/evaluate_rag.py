import json
import time
from datetime import datetime
from pathlib import Path

from app.services.chat import RAGService

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVALS_DIR / "golden_v2.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

REFUSALS = ["don't know", "do not know", "not mention", "doesn't mention",
            "doesn't cover", "does not cover", "no information"]

# Settings worth recording so runs can be compared
CONFIG_KEYS = ["chat_provider", "chat_model", "chat_temperature",
               "embed_provider", "embed_model", "chunk_size", "chunk_overlap",
               "top_k", "score_threshold", "system_prompt"]


def clean(text):
    return text.lower().replace(",", "").replace("’", "'")


def has(text, word):
    # "a|b" means a or b
    return any(option in text for option in word.split("|"))


def rate(results):
    return round(sum(results) / len(results), 3) if results else None


def pct(value):
    return f"{value:.0%}" if value is not None else "n/a"


def load_tests():
    with open(GOLDEN_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_test(rag, t):
    """Ask one golden question and grade the answer."""
    start = time.perf_counter()
    result = rag.ask(t["question"])
    record = {
        "id": t["id"],
        "type": t["type"],
        "question": t["question"],
        "answer": result.answer,
        "sources": [s.model_dump() for s in result.sources],
        "seconds": round(time.perf_counter() - start, 2),
    }
    answer = clean(result.answer)

    if t["type"] == "answerable":
        # What did the librarian actually hand the writer?
        chunks = rag.store.similarity_search(t["question"], k=rag.settings.top_k)
        context = clean(" ".join(c.page_content for c in chunks))

        record["page_hit"] = any(s.page in t["expected_pages"] for s in result.sources)
        record["fact_in_chunks"] = all(has(context, w) for w in t["must_include"])
        record["passed"] = all(has(answer, w) for w in t["must_include"])
        record["blame"] = ("generator: the fact WAS in the chunks" if record["fact_in_chunks"]
                           else "retriever: the fact was NOT in the chunks")
    elif t["type"] == "out_of_scope":
        record["passed"] = any(p in answer for p in REFUSALS)
        record["blame"] = "should have said it doesn't know"
    else:
        record["passed"] = not any(has(answer, w) for w in t["must_not_include"])
        record["blame"] = "leaked staff-only information"

    if record["passed"]:
        record.pop("blame")
    return record


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
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{started:%Y-%m-%d_%H-%M-%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path


def main():
    started = datetime.now()
    tests = load_tests()
    rag = RAGService()
    records = []

    try:
        for t in tests:
            record = run_test(rag, t)
            records.append(record)
            print("✓" if record["passed"] else "✗", t["id"])
            if not record["passed"]:
                print("   answer:", record["answer"])   # full answer, not cut off
                print("   blame: ", record["blame"])
    finally:
        rag.close()

    summary = summarize(records)
    print(f"\nPassed: {summary['passed']}/{summary['total']}")
    print("Retrieval hit rate (page):", pct(summary["retrieval_hit_rate"]))
    print("Fact in chunks (stricter): ", pct(summary["fact_in_chunks_rate"]))
    print("Answer correctness:        ", pct(summary["answer_correctness"]))
    print("Refusal rate:              ", pct(summary["refusal_rate"]))
    print("Leak rate:                 ", pct(summary["leak_rate"]))

    settings = rag.settings.model_dump()
    report = {
        "started_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round((datetime.now() - started).total_seconds(), 1),
        "config": {k: settings[k] for k in CONFIG_KEYS},
        "summary": summary,
        "results": records,
    }
    print("\nSaved:", save_report(report, started).relative_to(EVALS_DIR.parent))


if __name__ == "__main__":
    main()