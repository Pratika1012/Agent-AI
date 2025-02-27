import os
import streamlit as st
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        # ✅ Load Hugging Face sentence-transformer model
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Load Pinecone credentials from Streamlit secrets
        self.api_key = st.secrets["api_keys"].get("pinecone")
        self.environment = st.secrets["pinecone_config"].get("environment", "us-east-1")
        self.index_name = st.secrets["pinecone_config"].get("index_name", "ai-memory")

        # ✅ Ensure API key is available
        if not self.api_key:
            raise ValueError("❌ Pinecone API Key is missing! Check `.streamlit/secrets.toml`.")

        # ✅ Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.api_key)
            print("✅ Pinecone client initialized successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Error initializing Pinecone client: {e}")

        # ✅ Ensure the index exists before using it
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            print(f"⚠️ Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Match embedding model
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.environment)
            )

        # ✅ Correct way to load the index
        try:
            print(f"✅ Loading Pinecone index: {self.index_name}")
            
            # 🚀 Pass only index name (not Index object) to Langchain
            self.db = LangchainPinecone.from_existing_index(
                index_name=self.index_name,  # ✅ Correct way
                embedding=self.embed_model
            )
            print(f"✅ Pinecone index `{self.index_name}` successfully loaded!")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading Pinecone index `{self.index_name}`: {e}")

    def store_interaction(self, query: str, response: str):
        """
        Stores user queries and responses in Pinecone.
        """
        embedding = self.embed_model.embed_query(query)

        # ✅ Upsert the embedding into Pinecone with metadata
        self.db.add_texts(
            texts=[query],
            metadatas=[{"response": response}]
        )
        print(f"✅ Stored interaction: {query} -> {response}")

    def retrieve_similar(self, query: str, k: int = 2):
        """
        Retrieves past similar queries from Pinecone.
        """
        embedding = self.embed_model.embed_query(query)

        # ✅ Query Pinecone for similar vectors
        results = self.db.similarity_search_by_vector(embedding, k=k)

        # ✅ Extract responses from metadata
        return [res.metadata["response"] for res in results] if results else []

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone index.
        """
        self.db.delete(delete_all=True)
        print("🗑️ Cleared all interactions from Pinecone index.")
