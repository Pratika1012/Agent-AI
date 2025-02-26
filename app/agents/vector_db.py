import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ No API key required
from langchain.schema import Document
from langchain.docstore.document import Document as LangchainDocument

# ✅ Initialize Pinecone
PINECONE_API_KEY =st.secrets["api_keys"]["pinecone"]  # Set your Pinecone API key
PINECONE_ENV = "us-west1-gcp"  # Change this based on your Pinecone environment

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

class VectorDB:
    def __init__(self, index_name="memory-db"):
        """
        Initialize Pinecone vector storage with Hugging Face embeddings.
        """
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure Pinecone Index exists
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=384, metric="cosine")

        # ✅ Connect to Pinecone Index
        self.index = pinecone.Index(index_name)

        # ✅ Initialize LangChain Pinecone VectorStore
        self.db = Pinecone(self.index, self.embed_model.embed_query, "text")

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
