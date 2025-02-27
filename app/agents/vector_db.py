import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class VectorDB:
    def __init__(self):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        Loads API key, environment, and index name from Streamlit secrets.
        """
        # ✅ Load Pinecone config from Streamlit secrets
        api_key = st.secrets["api_keys"]["pinecone"]
        environment = st.secrets["pinecone_config"]["environment"]
        index_name = st.secrets["pinecone_config"]["index_name"]

        # ✅ Load Hugging Face sentence-transformer model
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Initialize Pinecone instance (New Method)
        self.pc = Pinecone(api_key=api_key)

        # ✅ Check if the index exists, else create it
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=384,  # Ensure this matches your embedding model's output dimensions
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment)
            )

        # ✅ Connect to the Pinecone index
        self.index = self.pc.Index(index_name)
        self.db = PineconeStore(self.index, self.embed_model)

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone for future recall.
        """
        embedding = self.embed_model.embed_query(query)
        vector_id = str(hash(query))  # Generate a unique ID for the query
        metadata = {"response": response}

        self.index.upsert(vectors=[(vector_id, embedding, metadata)])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries to provide context-aware responses.
        """
        embedding = self.embed_model.embed_query(query)
        results = self.index.query(vector=embedding, top_k=k, include_metadata=True)
        
        return [match["metadata"]["response"] for match in results["matches"] if "metadata" in match]

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone vector database.
        """
        self.index.delete(delete_all=True)
