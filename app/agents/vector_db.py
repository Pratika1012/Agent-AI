import os
import toml
import pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec
import streamlit as st
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings


# ✅ Load secrets from Streamlit's secrets management
secrets = st.secrets

# ✅ Retrieve API keys and Pinecone configuration
PINECONE_API_KEY = secrets["api_keys"]["pinecone"]
PINECONE_ENV = secrets["pinecone_config"]["environment"]
INDEX_NAME = secrets["pinecone_config"]["index_name"]

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


# ✅ Check if the index already exists before creating
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    spec = ServerlessSpec(cloud="aws", region="us-east-1")  # Change region if needed
    pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine", spec=spec)

# ✅ Connect to existing index
index = pc.Index(INDEX_NAME)

 # 384 is the embedding size for MiniLM

# ✅ Connect to Pinecone index


class VectorDB:
    def __init__(self):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Pinecone(index, self.embed_model, namespace="default")

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone.
        """
        embedding = self.embed_model.embed_query(query)
        index.upsert(vectors=[{"id": query, "values": embedding, "metadata": {"response": response}}])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries from Pinecone.
        """
        embedding = self.embed_model.embed_query(query)
        results = index.query(vector=embedding, top_k=k, include_metadata=True)
        return [match["metadata"]["response"] for match in results["matches"]]

    def clear_memory(self):
        """
        Clears all stored interactions in Pinecone.
        """
        index.delete(delete_all=True)

# ✅ Streamlit UI
