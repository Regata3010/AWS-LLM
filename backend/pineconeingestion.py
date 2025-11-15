import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

load_dotenv()

def create_pinecone_index():
    """Create Pinecone index if it doesn't exist"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "biasguard-regulations"
    
    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists. Deleting and recreating...")
        pc.delete_index(index_name)
        time.sleep(1)  # Wait for deletion
    
    # Create new index
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        )
    )
    
    print("Waiting for index to be ready...")
    time.sleep(5)
    print("Index created successfully!")

def load_regulations():
    """Load all regulation documents"""
    regulations_dir = Path("core/regulations")
    
    if not regulations_dir.exists():
        print(f"ERROR: {regulations_dir} not found!")
        print("Please create the directory and add regulation files.")
        return []
    
    documents = []
    
    # Define documents with metadata
    doc_configs = [
        {
            "filename": "cfpb_ecoa.txt",
            "metadata": {
                "source": "CFPB Regulation B",
                "regulation": "ECOA",
                "category": "lending",
                "year": "1974",
                "authority": "Consumer Financial Protection Bureau"
            }
        },
        {
            "filename": "eeoc_guidelines.txt",
            "metadata": {
                "source": "EEOC Uniform Guidelines",
                "regulation": "Title VII",
                "category": "employment",
                "year": "1978",
                "authority": "Equal Employment Opportunity Commission"
            }
        },
        {
            "filename": "griggs_v_duke_power.txt",
            "metadata": {
                "source": "U.S. Supreme Court",
                "regulation": "Disparate Impact Doctrine",
                "category": "case_law",
                "year": "1971",
                "case": "Griggs v. Duke Power Co.",
                "citation": "401 U.S. 424"
            }
        }
    ]
    
    for config in doc_configs:
        filepath = regulations_dir / config["filename"]
        
        if not filepath.exists():
            print(f"WARNING: {filepath} not found. Skipping...")
            continue
        
        print(f"\nLoading: {config['filename']}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"  - Size: {len(content)} characters")
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata=config["metadata"]
        )
        
        documents.append(doc)
    
    return documents

def chunk_documents(documents):
    """Split documents into chunks for embedding"""
    print("\nChunking documents...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     
        chunk_overlap=200,   
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Show sample
    if chunks:
        print(f"\nSample chunk:")
        print(f"  - Content preview: {chunks[0].page_content[:150]}...")
        print(f"  - Metadata: {chunks[0].metadata}")
    
    return chunks

def ingest_to_pinecone(chunks):
    """Embed and store chunks in Pinecone"""
    print("\nEmbedding and storing in Pinecone...")
    print("This may take 1-2 minutes...")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Store in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="biasguard-regulations"
    )
    
    print(f"Successfully ingested {len(chunks)} chunks into Pinecone!")
    
    return vectorstore

def test_retrieval(vectorstore):
    """Test that retrieval works"""
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    
    test_queries = [
        "What is the four-fifths rule?",
        "What is disparate impact?",
        "ECOA prohibited basis",
        "business necessity test"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        docs = vectorstore.similarity_search(query, k=2)
        
        if docs:
            print(f"  Found {len(docs)} relevant chunks:")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. Source: {doc.metadata.get('source')}")
                print(f"     Preview: {doc.page_content[:100]}...")
        else:
            print("  No results found!")

def main():
    """Main ingestion pipeline"""
    print("="*60)
    print("BiasGuard Regulation Ingestion Pipeline")
    print("="*60)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY not set in .env")
        return
    
    print("\nEnvironment variables: OK")
    
    # Step 1: Create index
    create_pinecone_index()
    
    # Step 2: Load documents
    documents = load_regulations()
    
    if not documents:
        print("\nERROR: No documents loaded. Check data/regulations/ folder.")
        return
    
    print(f"\nLoaded {len(documents)} documents successfully!")
    
    # Step 3: Chunk documents
    chunks = chunk_documents(documents)
    
    # Step 4: Ingest to Pinecone
    vectorstore = ingest_to_pinecone(chunks)
    
    # Step 5: Test retrieval
    test_retrieval(vectorstore)
    
    print("\n" + "="*60)
    print("INGESTION COMPLETE!")
    print("="*60)
    print(f"\nYour RAG system is ready with {len(chunks)} knowledge chunks.")
    print("You can now test the agent with: python scripts/test_agent.py")

if __name__ == "__main__":
    main() 