import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec  # ✅ Correct Pinecone import
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from pinecone import Pinecone

# ✅ Load API key from Streamlit secrets or environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", st.secrets.get("api_keys", {}).get("pinecone_api_key"))

if not PINECONE_API_KEY:
    st.error("❌ Pinecone API Key Not Found! Check secrets.toml or environment variables.")
    raise ValueError("Pinecone API key not found!")

# ✅ Correct Pinecone Client Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Load Pinecone API Key

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
