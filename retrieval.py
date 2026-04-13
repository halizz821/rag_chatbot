import os
import pickle
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

class HybridRetriever:
    def __init__(self, persist_dir: str = "db/chroma_db", bm25_path: str = "db/bm25_docs.pkl"):
        self.persist_dir = persist_dir
        self.bm25_path = bm25_path
        self.hybrid_retriever = None
        self.chroma_retriever = None
        self.bm25_retriever = None
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        print("Setting up Vector Retriever...")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Load Chroma Vector Store
        if os.path.exists(self.persist_dir):
            vectorstore = Chroma(
                collection_name="rag_ingestion",
                embedding_function=embedding_model,
                persist_directory=self.persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
            self.chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        else:
            print("Chroma persist directory not found. Initialize vectorstore first.")
            self.chroma_retriever = None

        print("Setting up BM25 Retriever...")
        # Load BM25 documents
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                documents = pickle.load(f)
            if documents:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = 2
        else:
            print("BM25 documents not found. Initialize ingestion first.")
            self.bm25_retriever = None

        if self.chroma_retriever and self.bm25_retriever:
            print("Setting up Hybrid Retriever...")
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[self.chroma_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]
            )
            print("Hybrid Setup complete!")
        else:
            print("Warning: Could not initialize full Hybrid Retriever. Please ingest documents first.")

    def reload(self):
        """Reload indices after a new document is ingested."""
        self._initialize_retrievers()

    def retrieve(self, query: str):
        """Invoke hybrid search"""
        if not self.hybrid_retriever:
            return []
        
        return self.hybrid_retriever.invoke(query)
