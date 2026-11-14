from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings # Embedding Model
from langchain_core.vectorstores import InMemoryVectorStore # Vector Database
from langchain.chat_models import init_chat_model # LLM


DATA_PATH = "data/shop_handbook.pdf"

CHAT_MODEL = "gemma3:1b"
CHAT_PROVIDER = "ollama"
CHAT_TEMPERATURE = 0.1

EMBED_MODEL = "nomic-embed-text:latest"
EMBED_PROVIDER = "ollama"
EMBED = 768

TOP_K = 3


# Indexing
# 1. Load
pages = PyPDFLoader(DATA_PATH).load() # List of Documents

# 2. Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(pages)

# 3. Embed
embed = init_embeddings(
    model=EMBED_MODEL,
    provider=EMBED_PROVIDER
)

# print(len(embed.embed_query("Hi")))

# 4. Store
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embed
)

# Answering
# LLM
llm = init_chat_model(
    model=CHAT_MODEL,
    model_provider=CHAT_PROVIDER,
    temperature=CHAT_TEMPERATURE
)

query = input("You: ").strip()

# 5. Retrieve
retrieved_docs = vector_store.similarity_search(query, k=TOP_K) # 3 docs

context = "\n\n".join(d.page_content for d in retrieved_docs)

SYSTEM = (
    "You are helpful Clause shop assistant. "
    "Answers only from provided context. "
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
]

# Augment and Generate
print(llm.invoke(messages).text)