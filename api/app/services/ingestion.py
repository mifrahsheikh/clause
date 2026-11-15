import re
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

from app.core.config import (
    get_embeddings,
    get_qdrant_client,
    get_settings,
)

STAFF_MARKER = re.compile(
    r"INTERNAL\s+STAFF\s+INFORMATION|STAFF\s+ONLY",
    re.IGNORECASE,
)

FORBIDDEN_PHRASES = [
    "manual fraud review",
    "extension 4410",
    "risk team",
    "orders above rs 50,000",
]


def load(path: Path):
    docs = PyPDFLoader(str(path)).load()

    filtered_docs = []
    inside_staff_section = False

    for doc in docs:
        safe_lines = []

        for line in doc.page_content.splitlines():
            if STAFF_MARKER.search(line):
                inside_staff_section = True
                continue

            if not inside_staff_section:
                safe_lines.append(line)

        doc.page_content = "\n".join(safe_lines)
        filtered_docs.append(doc)

    complete_text = "\n".join(
        doc.page_content for doc in filtered_docs
    ).lower()

    for phrase in FORBIDDEN_PHRASES:
        if phrase in complete_text:
            raise ValueError(
                f"SECURITY FAILURE: Restricted content found: {phrase}"
            )

    return filtered_docs


def ingest(path: str | None = None) -> int:

    settings = get_settings()

    file = Path(path or settings.data_path).resolve()

    if not file.exists():
        raise FileNotFoundError(file)

    print("Source PDF:", file)

    # 1. Load and filter.
    documents = load(file)

    # 2. Split into chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No safe chunks found.")

    # 3. Verify every chunk.
    for chunk in chunks:
        content = chunk.page_content.lower()

        for phrase in FORBIDDEN_PHRASES:
            if phrase in content:
                raise ValueError(
                    f"SECURITY FAILURE: {phrase} found in chunk."
                )

    print("Staff-only filtering: PASSED")
    print("Safe chunks prepared:", len(chunks))

    # 4. Initialize embeddings.
    embeddings = get_embeddings(settings)

    actual_dim = len(
        embeddings.embed_query("dimension check")
    )

    if actual_dim != settings.embed_dim:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"expected {settings.embed_dim}, got {actual_dim}"
        )

    # 5. Delete the old collection.
    client = get_qdrant_client(settings)

    try:
        if client.collection_exists(settings.collection_name):
            client.delete_collection(settings.collection_name)
            print("Old collection deletion requested.")

    finally:
        client.close()

    # 6. Remove stale local collection storage.
    if not settings.qdrant_url:

        storage_path = Path(settings.qdrant_path).resolve()

        collection_dir = (
            storage_path
            / "collection"
            / settings.collection_name
        )

        print("Qdrant storage:", storage_path)
        print("Collection storage:", collection_dir)

        if collection_dir.exists():
            shutil.rmtree(collection_dir)
            print("Old collection storage removed.")

        if collection_dir.exists():
            raise RuntimeError(
                "Could not remove old collection storage."
            )

    # 7. Reopen Qdrant.
    client = get_qdrant_client(settings)

    try:
        # 8. Create a fresh collection.
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embed_dim,
                distance=Distance.COSINE,
            ),
        )

        print("Fresh collection created.")

        # Verify it starts empty.
        initial_count = client.count(
            collection_name=settings.collection_name,
            exact=True,
        ).count

        if initial_count != 0:
            raise RuntimeError(
                f"Fresh collection is not empty: {initial_count} points."
            )

        # 9. Initialize vector store.
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embeddings,
        )

        # 10. Insert safe chunks.
        vector_store.add_documents(chunks)

        # 11. Verify indexing.
        actual_count = client.count(
            collection_name=settings.collection_name,
            exact=True,
        ).count

        print("\n========== INGESTION RESULTS ==========")
        print("Expected chunks:", len(chunks))
        print("Actual Qdrant points:", actual_count)

        if actual_count != len(chunks):
            raise RuntimeError(
                f"Indexing mismatch: expected {len(chunks)}, "
                f"found {actual_count}."
            )

        print("Staff-only filtering: PASSED")
        print("Qdrant indexing: PASSED")
        print("INGESTION COMPLETE")

        return actual_count

    finally:
        client.close()


if __name__ == "__main__":
    ingest()