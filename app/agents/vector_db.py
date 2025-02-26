import os
import pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

class VectorDB:
    def __init__(self, index_name: str, persist_directory=None):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.

        Args:
            index_name (str): Name of the Pinecone index.
            persist_directory (str, optional): Not used for Pinecone. Included for compatibility.
        """
        # ✅ Load Hugging Face sentence-transformer model
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Initialize Pinecone
        self.api_key = os.getenv("PINECONE_API_KEY") or st.secrets["api_keys"]["pinecone"]
        self.environment = os.getenv("PINECONE_ENV") or st.secrets["pinecone_config"]["environment"]
        self.index_name = index_name

        # ✅ Debug: Print API key and environment
        print(f"Pinecone API Key: {self.api_key}")
        print(f"Pinecone Environment: {self.environment}")

        # ✅ Ensure API key is available
        if not self.api_key:
            raise ValueError("❌ Pinecone API Key is missing! Set it in environment variables or Streamlit secrets.")

        # ✅ Initialize Pinecone client
        try:
            self.pc = pinecone.Pinecone(api_key=self.api_key)
            print("Pinecone initialized successfully!")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

        # ✅ Check if the index exists, create it if it doesn't
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Match the dimension of the embedding model
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

        Args:
            query (str): The user's query.
            response (str): The generated response.
        """
        # Generate embedding for the query
        embedding = self.embed_model.embed_query(query)

        # Upsert the embedding into Pinecone with metadata
        self.index.upsert(
            vectors=[
                {
                    "id": query,  # Use query as the ID (or generate a unique ID)
                    "values": embedding,
                    "metadata": {"response": response}
                }
            ]
        )
        print(f"Stored interaction: {query} -> {response}")

    def retrieve_similar(self, query: str, k: int = 2) -> list:
        """
        Retrieves past similar queries to provide context-aware responses.

        Args:
            query (str): The user's query.
            k (int): Number of similar results to retrieve.

        Returns:
            list: A list of similar responses.
        """
        # Generate embedding for the query
        embedding = self.embed_model.embed_query(query)

        # Query Pinecone for similar vectors
        results = self.index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True
        )

        # Extract responses from metadata
        if results.get("matches"):
            print(f"Retrieved similar interactions: {results['matches']}")
            return [match["metadata"]["response"] for match in results["matches"]]
        else:
            print("No similar interactions found.")
            return []

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone index.
        """
        # Delete all vectors in the index
        self.index.delete(delete_all=True)
        print("Cleared all interactions from Pinecone index.")
