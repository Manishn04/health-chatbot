import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found! Please check your .env file.")

# Initialize Pinecone client (v5+)
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_embeddings():
    """Return HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def download_embeddings(index_name="health-chatbot-index"):
    embeddings = get_embeddings()

    # Create index if it doesn't exist
    if index_name not in [index["name"] for index in pc.list_indexes()]:
        print(f"⚠️ Index '{index_name}' not found. Creating a new one...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # adjust if needed
        )
    else:
        print(f"✅ Using existing Pinecone index: {index_name}")

    # Connect to index with LangChain
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    return vectorstore.as_retriever()
