import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings  

# ✅ Ensure Pinecone API Key is Loaded Correctly
# ✅ Debug Pinecone API Key Retrieval
try:
    PINECONE_API_KEY = st.secrets["api_keys"]["pinecone"]
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API Key is empty.")
    st.success("✅ Pinecone API Key successfully loaded.")  # Debugging message
except KeyError:
    st.error("❌ Pinecone API key is missing in Streamlit secrets. Check secrets.toml!")
    raise ValueError("Pinecone API key not found!")

# ✅ Debugging API Key Output
st.write(f"Pinecone API Key (masked): {PINECONE_API_KEY[:5]}...")  # Print first 5 characters

INDEX_NAME = "ai-memory"

# ✅ Correct Pinecone Initialization
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1") 

class VectorDB:
    def __init__(self):
        """ Initialize Pinecone with LangChain """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure the index exists
        if INDEX_NAME not in [index['name'] for index in pc.list_indexes()]:
          # ✅ Fix: Use .names()
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
                  # ✅ Required for Pinecone v3
            )

        # ✅ Connect to the index
        self.index = pc.Index(INDEX_NAME)

        # ✅ Use PineconeVectorStore Correctly
        self.db = PineconeVectorStore.from_existing_index(INDEX_NAME, self.embed_model)

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
