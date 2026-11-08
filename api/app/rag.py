from langchain.chat_models import init_chat_model #llm
from langchain.embeddings import init_embeddings #embedding
# from langchain_community.document_loaders import TextLoader #py pdf loader
from langchain_core.documents import Document #.txt , .md
from langchain_text_splitters import RecursiveCharacterTextSplitter #chunks
from langchain_core.vectorstores import InMemoryVectorStore #vs
from pathlib import Path

#indexing
# 1. Loading the document
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
text=(DATA_DIR / "shop_handbook.md").read_text(encoding="utf-8")
doc= Document(page_content=text , metadata={"source":"shop_handbook.md"})

# 2 chunking the document --chunk size should lie in 500 to 1000 and chunk overlap should lie in 10% to 20%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, separators=["\n\n", '\n', '',''])

# 3. 
vector_store = InMemoryVectorStore(
    embedding=init_embeddings
)

