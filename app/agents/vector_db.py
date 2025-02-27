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
        try:
            self.api_key = st.secrets["api_keys"]["pinecone"]
            self.environment = st.secrets["pinecone_config"]["environment"]
            self.index_name = st.secrets["pinecone_config"]["index_name"]
        except KeyError as e:
            st.error(f"❌ Missing Pinecone secret: {e}")
            raise ValueError("Pinecone API Key or Environment not found in secrets.toml")

        # ✅ Initialize Pinecone client
        try:
            self.pc = pinecone.Pinecone(api_key=self.api_key)
            print("✅ Pinecone initialized successfully!")
        except Exception as e:
            raise RuntimeError(f"❌ Error initializing Pinecone: {e}")

        # ✅ Check if index exists, create if missing
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            print(f"⚠️ Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Match embedding model
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region=self.environment)
            )

        # ✅ Connect to the Pinecone index
        self.index = self.pc.Index(self.index_name)

        # ✅ Initialize Langchain Pinecone wrapper
        self.db = LangchainPinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embed_model
        )

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
