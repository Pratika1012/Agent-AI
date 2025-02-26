import os
import streamlit as st
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Load Pinecone API Key Correctly
PINECONE_API_KEY = st.secrets.get("pinecone_api_key", None)

# ✅ Fallback to Environment Variable
if not PINECONE_API_KEY:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    st.error("❌ Pinecone API Key is missing! Check Streamlit secrets or environment variables.")
    raise ValueError("Pinecone API Key Not Found!")

# ✅ Debugging
st.write(f"✅ Pinecone API Key Loaded: {PINECONE_API_KEY[:5]}...")
st.write(f"✅ Pinecone Version: {pinecone.__version__}")

# ✅ Correct Pinecone Client Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "ai-memory"

class VectorDB:
    def __init__(self):
        """ Initialize Pinecone with LangChain """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure the index exists
        existing_indexes = [index["name"] for index in pc.list_indexes()]
        st.write(f"📌 Available Pinecone Indexes: {existing_indexes}")

        if INDEX_NAME not in existing_indexes:
            st.warning(f"🛑 Index '{INDEX_NAME}' not found. Creating a new one...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            st.success(f"✅ Index '{INDEX_NAME}' created successfully!")

        # ✅ Connect to the index manually
        self.index = pc.Index(INDEX_NAME)
        st.success(f"✅ Successfully connected to index '{INDEX_NAME}'!")

        # 🔥 FIX: Explicitly pass Pinecone Client and API Key to LangChain
        try:
            self.db = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=self.embed_model,
                api_key=PINECONE_API_KEY,  # ✅ FIX: Pass API Key Explicitly
                environment="us-east-1",  # ✅ FIX: Pass Environment Explicitly
                text_key="text",
                namespace=""
            )
            st.success("✅ Pinecone Vector Store Initialized Successfully!")

        except Exception as e:
            st.error(f"❌ Error Initializing Pinecone Vector Store: {str(e)}")
            raise e

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
