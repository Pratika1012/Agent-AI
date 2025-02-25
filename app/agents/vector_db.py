import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document

class VectorDB:
    def __init__(self, pinecone_api_key, environment="us-east-1", index_name="ai-memory"):
        """
        Initialize Pinecone vector database for memory storage.
        """
        self.embed_model = OpenAIEmbeddings()  # Using OpenAI embeddings

        # ✅ Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment=environment)

        # ✅ Connect to Pinecone index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=1536)  # Use OpenAI's embedding dimension

        self.db = Pinecone(index_name=index_name, embedding_function=self.embed_model)

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in Pinecone for future recall.
        """
        doc = Document(page_content=query, metadata={"response": response})
        self.db.add_texts([query], metadatas=[{"response": response}])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries to provide context-aware responses.
        """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]
