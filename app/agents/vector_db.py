import os
import streamlit as st
import pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Debugging: Check if API Key is Retrieved
PINECONE_API_KEY = st.secrets.get("api_keys", {}).get("pinecone", None)

# ✅ If Streamlit Secrets fail, use Environment Variable as Backup
if not PINECONE_API_KEY:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Debugging Output (Only show first 5 characters for security)
if PINECONE_API_KEY:
    st.write(f"✅ Pinecone API Key Loaded: {PINECONE_API_KEY[:5]}...")
else:
    st.error("❌ Pinecone API Key is missing! Check Streamlit secrets or set it as an environment variable.")
    raise ValueError("Pinecone API Key Not Found!")

# ✅ Initialize Pinecone Properly
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")

INDEX_NAME = "ai-memory"

class VectorDB:
    def __init__(self):
        """ Initialize Pinecone with LangChain """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure the index exists
        existing_indexes = pinecone.list_indexes()
        st.write(f"📌 Available Pinecone Indexes: {existing_indexes}")  # ✅ Debugging Index List

        if INDEX_NAME not in existing_indexes:
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ Required for Pinecone v3
            )

        # ✅ Connect to the index
        self.index = pinecone.Index(INDEX_NAME)

        # ✅ Use PineconeVectorStore Correctly
        self.db = PineconeVectorStore.from_existing_index(INDEX_NAME, self.embed_model)

    def store_interaction(self, query, response):
        """ Store query and response in Pinecone """
        self.db.add_texts([query], metadatas=[{"response": response}])

    def retrieve_similar(self, query, k=2):
        """ Retrieve similar queries """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]

    def clear_memory(self):
        """ Clear all stored interactions """
        self.index.delete(delete_all=True)
