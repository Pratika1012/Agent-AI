import os
import streamlit as st
import asyncio
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# ✅ Ensure an event loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load secrets from Streamlit Secrets Management & Ensure API key is set
secrets = st.secrets
PINECONE_API_KEY = secrets["api_keys"].get("pinecone", os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = secrets["pinecone_config"]["environment"]
INDEX_NAME = secrets["pinecone_config"]["index_name"]

# ✅ Ensure API key is available before initializing Pinecone
if not PINECONE_API_KEY:
    raise ValueError("❌ Pinecone API Key is missing! Set it in Streamlit secrets or environment variables.")

# ✅ Set API key in environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# ✅ Check if the index exists before creating
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    spec = ServerlessSpec(cloud="aws", region="us-east-1")  # Change region if needed
    pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine", spec=spec)

# ✅ Connect to the Pinecone index
index = pc.Index(INDEX_NAME)

class VectorDB:
    def __init__(self):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        try:
            self.db = LangchainPinecone.from_existing_index(
                index_name=INDEX_NAME, 
                embedding=self.embed_model, 
                pinecone_api_key=PINECONE_API_KEY,
                environment=PINECONE_ENV
            )
        except Exception as e:
            logging.error(f"Error initializing Pinecone: {e}")
            raise

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
        
        if results.get("matches"):
            return [match["metadata"]["response"] for match in results["matches"]]
        else:
            return ["No similar responses found."]

    def clear_memory(self):
        """
        Clears all stored interactions in Pinecone.
        """
        index.delete(delete_all=True)
