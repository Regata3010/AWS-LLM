# backend/core/ai/rag.py
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from pinecone import Pinecone, ServerlessSpec
import os
import json

class RegulationRAG:
    """RAG system using direct Pinecone API"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.index_name = "biasguard-regulations"
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get index (new Pinecone v7 API)
        self.index = self.pc.Index(self.index_name)
    
    def search_sync(self, query: str, k: int = 3):
        """Direct Pinecone search"""
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        formatted_results = []
        for match in results.get('matches', []):
            formatted_results.append({
                "content": match['metadata'].get('text', ''),
                "source": match['metadata'].get('source', 'Unknown'),
                "regulation": match['metadata'].get('regulation', ''),
                "relevance_score": round(match['score'], 3)
            })
        
        return formatted_results

# Singleton
_rag_instance = None

def get_rag_instance():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RegulationRAG()
    return _rag_instance

@tool
def search_regulations(query: str) -> str:
    """Search regulations using vector similarity"""
    rag = get_rag_instance()
    results = rag.search_sync(query, k=3)
    return json.dumps(results, indent=2)

@tool
def get_regulation_details(regulation_code: str) -> str:
    """Get full details of a specific regulation"""
    rag = get_rag_instance()
    query = f"{regulation_code} full text requirements"
    results = rag.search_sync(query, k=2)
    
    if not results:
        return json.dumps({"error": f"Regulation {regulation_code} not found"})
    
    return json.dumps({
        "regulation_code": regulation_code,
        "content": results[0]["content"],
        "source": results[0]["source"]
    }, indent=2)

def get_rag_tools():
    return [search_regulations, get_regulation_details]