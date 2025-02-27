import os
import streamlit as st
import pinecone
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

        # ✅ Explicitly set the Pinecone API Key in environment variables
        os.environ["PINECONE_API_KEY"] = self.api_key

        # ✅ Initialize Pinecone client properly
        try:
            self.pc = pinecone.Pinecone(api_key=self.api_key)
            print("✅ Pinecone initialized successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Error initializing Pinecone: {e}")

        # ✅ Ensure the index exists before using `from_existing_index`
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            print(f"⚠️ Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Match embedding model
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region=self.environment)
            )

        # ✅ Now, safely load the existing index
        try:
            self.index = self.pc.Index(self.index_name)
            self.db = LangchainPinecone(
                index=self.index,  # ✅ Pass Pinecone index object directly
                embedding=self.embed_model
            )
        except Exception as e:
            raise RuntimeError(f"❌ Error loading Pinecone index `{self.index_name}`: {e}")

    def store_interaction(self, query: str, response: str):
        """
        Stores user queries and responses in Pinecone.
        """
        embedding = self.embed_model.embed_query(query)

        # ✅ Upsert the embedding into Pinecone with metadata
        self.index.upsert([
            {
                "id": query,
                "values": embedding,
                "metadata": {"response": response}
            }
        ])
        print(f"✅ Stored interaction: {query} -> {response}")

    def retrieve_similar(self, query: str, k: int = 2):
        """
        Retrieves past similar queries from Pinecone.
        """
        embedding = self.embed_model.embed_query(query)

        # ✅ Query Pinecone for similar vectors
        results = self.index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True
        )

        # ✅ Extract responses from metadata
        if results.get("matches"):
            return [match["metadata"]["response"] for match in results["matches"]]
        else:
            print("⚠️ No similar interactions found.")
            return []

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone index.
        """
        self.index.delete(delete_all=True)
        print("🗑️ Cleared all interactions from Pinecone index.")
