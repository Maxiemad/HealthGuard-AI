from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class MedicalKnowledgeRetriever:
    def __init__(self):
        """Initialize the medical knowledge retriever"""
        self.index = None
        self.docs = []
        self.metadata = []
        self.model = None
        
        self._load_index()
    
    def _load_index(self):
        """Load the FAISS index and documents"""
        try:
            # Load FAISS index
            self.index = faiss.read_index('rag/indexes/medical_knowledge.index')
            
            # Load documents and metadata
            with open('rag/indexes/documents.pkl', 'rb') as f:
                data = pickle.load(f)
                self.docs = data['docs']
                self.metadata = data['metadata']
            
            # Load embedding model
            with open('rag/indexes/model_name.txt') as f:
                model_name = f.read().strip()
            self.model = SentenceTransformer(model_name)
            
            print(f"Loaded {len(self.docs)} document chunks")
            
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Please run rag/build_index.py first")
    
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant medical knowledge for a given query"""
        if self.model is None:
            return "Knowledge base not available"
        
        try:
            # Embed query
            query_embedding = self.model.encode([query])
            
            # Search index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.docs):
                    source = self.metadata[idx]['source']
                    chunk_text = self.docs[idx]
                    result = f"[{source.replace('.txt', '').title()}]\n{chunk_text}"
                    results.append(result)
            
            return "\n\n---\n\n".join(results)
            
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            return "Error retrieving medical knowledge"
    
    def get_relevant_sources(self, query: str, disease: str = None) -> list:
        """Get relevant sources for a specific disease or general query"""
        if disease:
            # Disease-specific query
            query = f"{disease} prevention guidelines {query}"
        
        context = self.retrieve(query, top_k=5)
        
        # Split into individual sources
        sources = []
        sections = context.split("\n\n---\n\n")
        for section in sections:
            if section.strip():
                sources.append(section)
        
        return sources

# Global retriever instance
_retriever = None

def get_retriever():
    """Get or create the global retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = MedicalKnowledgeRetriever()
    return _retriever

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Convenience function to retrieve context"""
    retriever = get_retriever()
    return retriever.retrieve(query, top_k)

if __name__ == "__main__":
    # Test the retriever
    retriever = get_retriever()
    
    # Test queries
    queries = [
        "diabetes prevention strategies",
        "heart disease risk factors",
        "kidney disease dietary recommendations",
        "blood pressure management"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = retriever.retrieve(query, top_k=2)
        print(results[:500] + "..." if len(results) > 500 else results)
        print()
