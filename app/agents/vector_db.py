import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ No API key required
from langchain.schema import Document
from langchain.docstore.document import Document as LangchainDocument

# ✅ Get Pinecone API Key
PINECONE_API_KEY = st.secrets.get("api_keys", {}).get("pinecone") or os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"  # Change this based on your Pinecone environment
INDEX_NAME = "ai-memory"

if not PINECONE_API_KEY:
    raise ValueError("🚨 Pinecone API Key not found! Ensure it's set in Streamlit secrets or environment variables.")

# ✅ Initialize Pinecone Client with API Key
pc = Pinecone(api_key=PINECONE_API_KEY)

class VectorDB:
    def __init__(self, index_name="ai-memory"):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure Pinecone Index Exists Before Connecting
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),  # ✅ Required for Pinecone v3
            )

        # ✅ Now Connect to Pinecone Index
        self.index = pc.Index(index_name)

        # ✅ Initialize LangChain PineconeVectorStore
        self.db = PineconeVectorStore.from_existing_index(
            index_name,
            self.embed_model,
        )  # ✅ Corrected usage

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone for future recall.
        """
        doc = LangchainDocument(page_content=query, metadata={"response": response})
        self.db.add_texts([query], metadatas=[{"response": response}])  # ✅ Corrected method

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries to provide context-aware responses.
        """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone index.
        """
        self.index.delete(delete_all=True)
