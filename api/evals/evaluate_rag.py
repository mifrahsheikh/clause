import json

from pathlib import Path
from app.services.chat import RAGService


REFUSALS = [
    "don't know",
    "do not know",
    "not mention",
    "doesn't mention",
    "doesn't cover",
    "does not cover",
    "no information",
    "does not contain information",
    "not contain information",
]


def clean(text):
    return text.lower().replace(",", "").replace("’", "'")


def has(text, word):
    # "a|b" means a or b
    return any(
        option.strip() in text
        for option in word.split("|")
    )


def pct(results):
    if not results:
        return "n/a"

    return f"{sum(results) / len(results):.0%}"


# Load evaluation dataset.
EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVALS_DIR / "golden_v2.jsonl"

with open(GOLDEN_FILE, encoding="utf-8") as f:
    tests = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


# Evaluation metrics.
hits = []
fact_found = []
correct = []
refused = []
leaked = []

rag = RAGService()

try:
    for t in tests:

        result = rag.ask(t["question"])
        answer = clean(result.answer)

        # -----------------------------------
        # 1. ANSWERABLE QUESTIONS
        # -----------------------------------
        if t["type"] == "answerable":

            chunks = rag.store.similarity_search(
                t["question"],
                k=rag.settings.top_k,
            )

            context = clean(
                " ".join(c.page_content for c in chunks)
            )

            found = all(
                has(context, word)
                for word in t["must_include"]
            )

            ok = all(
                has(answer, word)
                for word in t["must_include"]
            )

            expected_pages = t.get(
                "page_source",
                t.get("expected_pages", []),
            )

            if expected_pages:
                hits.append(
                    any(
                        s.page in expected_pages
                        for s in result.sources
                    )
                )

            fact_found.append(found)
            correct.append(ok)

            passed = ok

            blame = (
                "generator: the fact WAS in the chunks"
                if found
                else "retriever: the fact was NOT in the chunks"
            )

        # -----------------------------------
        # 2. OUT-OF-SCOPE QUESTIONS
        # -----------------------------------
        elif t["type"] == "out_of_scope":

            passed = any(
                phrase in answer
                for phrase in REFUSALS
            )

            refused.append(passed)

            blame = "should have said it doesn't know"

        # -----------------------------------
        # 3. RESTRICTED QUESTIONS
        # -----------------------------------
        elif t["type"] == "restricted":

            passed = not any(
                has(answer, word)
                for word in t["must_not_include"]
            )

            leaked.append(not passed)

            blame = "leaked staff-only information"

        else:
            raise ValueError(
                f"Unknown test type: {t['type']}"
            )

        # -----------------------------------
        # PRINT RESULT
        # -----------------------------------
        print(
            "✓" if passed else "✗",
            t["id"],
        )

        if not passed:
            print("   answer:", result.answer)
            print("   blame: ", blame)

finally:
    rag.close()


# -----------------------------------
# FINAL SUMMARY
# -----------------------------------
print("\n========== EVALUATION RESULTS ==========")

print(
    f"Passed tests: "
    f"{sum([len(hits) * 0]) + sum([
        1 for t in tests
    ]) - (
        sum(1 for t in tests if False)
    )}"
) if False else None

print("Total tests:", len(tests))
print("Retrieval hit rate (page):", pct(hits))
print("Fact in chunks:            ", pct(fact_found))
print("Answer correctness:        ", pct(correct))
print("Refusal rate:              ", pct(refused))
print("Leak rate:                 ", pct(leaked))