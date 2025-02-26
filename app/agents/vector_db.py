import os
import streamlit as st
import pinecone  # ✅ Correct Import
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ No API key required
from langchain.schema import Document
from langchain.docstore.document import Document as LangchainDocument

# ✅ Get Pinecone API Key from Streamlit Secrets
PINECONE_API_KEY = st.secrets["api_keys"]["pinecone"]
PINECONE_ENV = "us-east-1"  # Change this based on your Pinecone environment
INDEX_NAME = "ai-memory"

# ✅ Initialize Pinecone Correctly
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

class VectorDB:
    def __init__(self, index_name="ai-memory"):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure Pinecone Index Exists Before Connecting
        existing_indexes = pinecone.list_indexes()
        
        if index_name not in existing_indexes:
            pinecone.create_index(name=index_name, dimension=384, metric="cosine")

        # ✅ Now Connect to Pinecone Index
        self.index = pinecone.Index(index_name)  # ✅ Correct way to initialize Pinecone Index

        # ✅ Initialize LangChain Pinecone VectorStore with correct embedding function
        self.db = PineconeVectorStore.from_pinecone_index(
            self.index,
            self.embed_model,
            "text"
        )  # ✅ Corrected usage

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone for future recall.
        """
        doc = LangchainDocument(page_content=query, metadata={"response": response})
        self.db.add_documents([doc])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries to provide context-aware responses.
        """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]

    def clear_memory(self):
        """
        Clears all stored interactions in the Pinecone index.
        """
        self.index.delete(delete_all=True)
