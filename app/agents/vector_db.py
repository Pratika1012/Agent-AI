import os
import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings  

# ✅ Get Pinecone API Key from Streamlit secrets or environment
PINECONE_API_KEY = st.secrets["api_keys"]["pinecone"]
PINECONE_ENV = "us-east-1" 
INDEX_NAME = "ai-memory"

# ✅ Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY,PINECONE_ENV)

class VectorDB:
    def __init__(self):
        """ Initialize Pinecone with LangChain """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure the index exists
        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(INDEX_NAME, dimension=384, metric="cosine")

        # ✅ Connect to the index
        self.index = pinecone.Index(INDEX_NAME)

        # ✅ Use PineconeVectorStore
        self.db = PineconeVectorStore(self.index, self.embed_model.embed_query, "text")

    def store_interaction(self, query, response):
        """ Store query and response in Pinecone """
        self.db.add_texts([query], metadatas=[{"response": response}])

    def retrieve_similar(self, query, k=2):
        """ Retrieve similar queries """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]

    def clear_memory(self):
        """ Clear all stored interactions """
        self.index.delete(delete_all=True)
