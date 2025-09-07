
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "health-chatbot-index"

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found! Please check your .env file.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def load_pdfs(data_dir="./data"):
    """Load PDF documents from a directory."""
    print(f"📂 Loading PDFs from: {data_dir}")
    loader = PyPDFDirectoryLoader(data_dir)
    return loader.load()

def split_documents(documents):
    """Split large documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def create_index(data_dir="./data"):
    """Load PDFs, create embeddings, and upload to Pinecone index."""
    docs = load_pdfs(data_dir)
    print(f"📄 Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create index if it does not exist
    if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
        print(f"🆕 Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # embedding size for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust if needed
        )
    else:
        print(f"✅ Using existing Pinecone index: {INDEX_NAME}")

    print("📥 Uploading embeddings to Pinecone...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    print("✅ Pinecone index is ready!")

if __name__ == "__main__":
    create_index() 

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "health-chatbot-index"

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found! Please check your .env file.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def load_pdfs(data_dir="./data"):
    """Load PDF documents from a directory."""
    print(f"📂 Loading PDFs from: {data_dir}")
    loader = PyPDFDirectoryLoader(data_dir)
    return loader.load()

def split_documents(documents):
    """Split large documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def create_index(data_dir="./data"):
    """Load PDFs, create embeddings, and upload to Pinecone index."""
    docs = load_pdfs(data_dir)
    print(f"📄 Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create index if it does not exist
    if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
        print(f"🆕 Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # embedding size for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust if needed
        )
    else:
        print(f"✅ Using existing Pinecone index: {INDEX_NAME}")

    print("📥 Uploading embeddings to Pinecone...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    print("✅ Pinecone index is ready!")

if __name__ == "__main__":
    create_index() 

