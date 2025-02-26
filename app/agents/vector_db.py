import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec  # ✅ Correct Pinecone import
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Load Pinecone API Key
PINECONE_API_KEY = st.secrets.get("api_keys", {}).get("pinecone", None)

# ✅ Fallback to Environment Variable (For Streamlit Deployment)
if not PINECONE_API_KEY:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ✅ Ensure API Key is Present
if not PINECONE_API_KEY:
    st.error("❌ Pinecone API Key is missing! Check Streamlit secrets or environment variables.")
    raise ValueError("Pinecone API Key Not Found! Ensure it is set in Streamlit secrets or as an environment variable.")

st.write(f"✅ Pinecone API Key Loaded: {PINECONE_API_KEY[:5]}...")  # ✅ Debugging Output

# ✅ Correct Pinecone Client Initialization (Fixed)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "ai-memory"

class VectorDB:
    def __init__(self):
        """ Initialize Pinecone with LangChain """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure the index exists
        existing_indexes = pc.list_indexes().names()  # ✅ Fix: Use .names() correctly
        st.write(f"📌 Available Pinecone Indexes: {existing_indexes}")  # ✅ Debugging Output

        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ Required for Pinecone v3
            )

        # ✅ Connect to the index
        self.index = pc.Index(INDEX_NAME)

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
