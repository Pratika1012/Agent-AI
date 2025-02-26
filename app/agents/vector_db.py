import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings  

# ✅ Ensure Pinecone API Key is Loaded Correctly
try:
    PINECONE_API_KEY = st.secrets["api_keys"]["pinecone"]
except KeyError:
    st.error("❌ Pinecone API key is missing in Streamlit secrets. Check secrets.toml!")
    raise ValueError("Pinecone API key not found!")

if not PINECONE_API_KEY:
    raise ValueError("❌ Pinecone API Key Not Found! Check Streamlit secrets.")

INDEX_NAME = "ai-memory"

# ✅ Initialize Pinecone Client (Remove pinecone.init)
pc = Pinecone(api_key=PINECONE_API_KEY)

class VectorDB:
    def __init__(self):
        """ Initialize Pinecone with LangChain """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure the index exists
        if INDEX_NAME not in pc.list_indexes().names():  # ✅ Fix: Use .names()
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ Required for Pinecone v3
            )

        # ✅ Connect to the index
        self.index = pc.Index(INDEX_NAME)

        # ✅ Use PineconeVectorStore
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
