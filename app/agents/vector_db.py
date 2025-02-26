import os
import pinecone  # ✅ Import Pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ No API key required
from langchain.schema import Document
from pinecone import Pinecone as PineconeClient

class VectorDB:
    def __init__(self, index_name="my_vector_index"):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        # ✅ Load Hugging Face sentence-transformer model
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Pinecone API Configuration (Replace with your API Key)
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Store API key in env variable
        PINECONE_ENV = "us-west1-gcp"  # Update with your Pinecone environment

        if not PINECONE_API_KEY:
            raise ValueError("❌ Pinecone API Key is missing! Set it as an environment variable.")

        # ✅ Initialize Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

        # ✅ Check if the index exists, else create one
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=384)  # 384 is the embedding size for MiniLM

        # ✅ Connect to the Pinecone index
        self.index = pinecone.Index(index_name)
        self.db = Pinecone(self.index, self.embed_model, namespace="default")

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone for future recall.
        """
        # ✅ Generate embedding for the query
        embedding = self.embed_model.embed_query(query)

        # ✅ Upsert (store) data into Pinecone
        self.index.upsert(vectors=[{"id": query, "values": embedding, "metadata": {"response": response}}])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries from Pinecone.
        """
        embedding = self.embed_model.embed_query(query)

        # ✅ Perform similarity search
        results = self.index.query(vector=embedding, top_k=k, include_metadata=True)

        # ✅ Return stored responses
        return [match["metadata"]["response"] for match in results["matches"]]

    def clear_memory(self):
        """
        Clears all stored interactions in Pinecone.
        """
        self.index.delete(delete_all=True)

