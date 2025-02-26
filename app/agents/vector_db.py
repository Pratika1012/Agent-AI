import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load secrets
secrets = st.secrets
PINECONE_API_KEY = secrets["api_keys"].get("pinecone", os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = secrets["pinecone_config"]["environment"]
INDEX_NAME = secrets["pinecone_config"]["index_name"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Check if index exists
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine", spec=spec)

# Connect to the index
index = pc.Index(INDEX_NAME)

# Initialize embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Store interaction
def store_interaction(query, response):
    embedding = embed_model.encode(query)
    index.upsert(vectors=[{"id": query, "values": embedding.tolist(), "metadata": {"response": response}}])

# Retrieve similar interactions
def retrieve_similar(query, k=2):
    embedding = embed_model.encode(query)
    results = index.query(vector=embedding.tolist(), top_k=k, include_metadata=True)
    if results.get("matches"):
        return [match["metadata"]["response"] for match in results["matches"]]
    else:
        return ["No similar responses found."]

# Clear memory
def clear_memory():
    index.delete(delete_all=True)
