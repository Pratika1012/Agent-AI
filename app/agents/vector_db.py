import os
import toml
import pinecone
import streamlit as st
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings

# ✅ Load secrets from Streamlit's secrets management
secrets = st.secrets

# ✅ Retrieve API keys and Pinecone configuration
PINECONE_API_KEY = secrets["api_keys"]["pinecone"]
PINECONE_ENV = secrets["pinecone_config"]["environment"]
INDEX_NAME = secrets["pinecone_config"]["index_name"]

# ✅ Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# ✅ Check if the index exists, else create one
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=384)  # 384 is the embedding size for MiniLM

# ✅ Connect to Pinecone index
index = pinecone.Index(INDEX_NAME)

class VectorDB:
    def __init__(self):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Pinecone(index, self.embed_model, namespace="default")

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone.
        """
        embedding = self.embed_model.embed_query(query)
        index.upsert(vectors=[{"id": query, "values": embedding, "metadata": {"response": response}}])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries from Pinecone.
        """
        embedding = self.embed_model.embed_query(query)
        results = index.query(vector=embedding, top_k=k, include_metadata=True)
        return [match["metadata"]["response"] for match in results["matches"]]

    def clear_memory(self):
        """
        Clears all stored interactions in Pinecone.
        """
        index.delete(delete_all=True)

# ✅ Streamlit UI
st.title("🔍 Pinecone-Powered Vector Search")

# Initialize the vector database
vector_db = VectorDB()

# User input
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        results = vector_db.retrieve_similar(query, k=2)
        if results:
            st.write("### Similar Responses:")
            for res in results:
                st.write(f"- {res}")
        else:
            st.warning("No similar results found.")
    else:
        st.error("Please enter a query.")

if st.button("Clear Memory"):
    vector_db.clear_memory()
    st.success("All stored interactions have been cleared.")
