from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DenseRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents):
        """Ingest a list of text documents."""
        self.documents = documents
        print("Encoding documents...")
        embeddings = self.encoder.encode(documents, convert_to_numpy=True)
        
        # Initialize FAISS (L2 Distance)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Index built with {len(documents)} documents.")

    def retrieve(self, query, k=3):
        """Returns top-k documents for a query."""
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, k)
        
        results = [self.documents[i] for i in indices[0]]
        return results