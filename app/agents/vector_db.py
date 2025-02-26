import os
import streamlit as st
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

class VectorDB:
    def __init__(self, index_name="ai-memory", environment="us-east-1"):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        # ✅ Load Pinecone API key securely
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", st.secrets.get("api_keys", {}).get("pinecone_api_key"))
        
        if not PINECONE_API_KEY:
            st.error("❌ Pinecone API Key Not Found! Check secrets.toml or environment variables.")
            raise ValueError("Pinecone API key not found!")

        # ✅ Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # ✅ Load Hugging Face sentence-transformer model
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Connect to Pinecone index
        self.index_name = index_name
        if self.index_name not in self.pc.list_indexes():
            self.pc.create_index(name=self.index_name, dimension=384, metric="cosine")  # Adjust dimension if needed
        
        # ✅ Initialize LangChain Pinecone vector store
        self.db = PineconeStore.from_existing_index(index_name=self.index_name, embedding=self.embed_model)

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone.
        """
        self.db.add_texts([query], metadatas=[{"response": response}])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries from Pinecone.
        """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone index.
        """
        self.pc.delete_index(self.index_name)
        self.pc.create_index(name=self.index_name, dimension=384, metric="cosine")
