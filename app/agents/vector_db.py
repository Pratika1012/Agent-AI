import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        Loads API key, environment, and index name from Streamlit secrets.
        """
        # ✅ Fetch Pinecone API Key with Fallback
        api_key = st.secrets["api_keys"].get("pinecone", os.getenv("PINECONE_API_KEY"))
        if not api_key:
            raise RuntimeError("❌ PINECONE_API_KEY is missing from secrets or environment variables!")

        # ✅ Set API key as environment variable (Fix for Pinecone SDK)
        os.environ["PINECONE_API_KEY"] = api_key

        # ✅ Load Pinecone Configuration
        environment = st.secrets["pinecone_config"].get("environment", "us-east-1")
        index_name = st.secrets["pinecone_config"].get("index_name", "ai-memory")

        # ✅ Load Hugging Face sentence-transformer model
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Initialize Pinecone with Explicit API Key
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        # ✅ Ensure Index Exists
        existing_indexes = self.pc.list_indexes().names()
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment)
            )

        # ✅ Connect to the Pinecone Index
        self.index = self.pc.Index(index_name)

        # ✅ Corrected: Initialize Pinecone Vector Store
        self.db = PineconeStore.from_existing_index(
            index_name=index_name,
            embedding=self.embed_model
        )

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone for future recall.
        """
        metadata = {"response": response, "text": query}
        self.db.add_texts([query], metadatas=[metadata])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries to provide context-aware responses.
        """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone vector database.
        """
        self.pc.delete_index(st.secrets["pinecone_config"]["index_name"])
