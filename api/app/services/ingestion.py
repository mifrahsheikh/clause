from pathlib import Path
import re
# from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
import re
from app.core.config import get_embeddings, get_qdrant_client, get_settings

from app.rag.cleaning import internal_section

STAFF_MARKER = re.compile(
    r"INTERNAL\s+STAFF\s+INFORMATION|STAFF\s+ONLY",
    re.IGNORECASE,
)

# Restricted information that must never enter Qdrant.
FORBIDDEN_PHRASES = [
    "manual fraud review",
    "extension 4410",
    "risk team",
    "orders above rs 50,000",
]

def load(path: Path) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()

    filtered_docs = []
    inside_staff_section = False

    for doc in docs:
        safe_lines = []

        for line in doc.page_content.splitlines():

            # Detect the beginning of restricted information.
            if STAFF_MARKER.search(line):
                inside_staff_section = True
                continue

            # Keep only customer-accessible information.
            if not inside_staff_section:
                safe_lines.append(line)

        filtered_docs.append(
            Document(
                page_content="\n".join(safe_lines),
                metadata=doc.metadata,
            )
        )

    # Verify that restricted information was removed.
    complete_text = "\n".join(
        doc.page_content for doc in filtered_docs
    ).lower()

    for phrase in FORBIDDEN_PHRASES:
        if phrase in complete_text:
            raise ValueError(
                f"SECURITY FAILURE: Restricted information found: {phrase}"
            )

    return filtered_docs

def ingest(path: str | None = None) -> int:
    settings = get_settings()

    # 1. Load
    file = Path(path or settings.data_path)
    documents = load(file)

    # 2. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError(f"No text found in {file}")

    # 3. Embed: check the model works before the old collection is deleted
    embed = get_embeddings(settings)

    actual_dim = len(embed.embed_query("dimension check"))
    if actual_dim != settings.embed_dim:
        raise ValueError(
            f"RAG_EMBED_DIM is {settings.embed_dim}, "
            f"but {settings.embed_model} returns {actual_dim} dimensions"
        )

    # 4. Store
    client = get_qdrant_client(settings)

    try:
        if client.collection_exists(settings.collection_name):
            client.delete_collection(settings.collection_name)

        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={"dense": VectorParams(size=settings.embed_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embed,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse"
        )
        vector_store.add_documents(chunks)
    finally:
        client.close()

    return len(chunks)


if __name__ == "__main__":
    print(ingest(), "chunks indexed")