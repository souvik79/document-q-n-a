"""
Test script for vector store (without Ollama dependency for now).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain.schema import Document

def main():
    print("=" * 60)
    print("Vector Store Test (Basic)")
    print("=" * 60)
    print()

    # Check if Ollama is running first
    print("Checking Ollama availability...")
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get('models', [])
            print(f"  Available models: {[m['name'] for m in models]}")
        else:
            print("✗ Ollama is not responding correctly")
            print("  Please start Ollama with: ollama serve")
            print("  And ensure llama3 is available: ollama pull llama3")
            return
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Please start Ollama with: ollama serve")
        print("  And ensure llama3 is available: ollama pull llama3")
        return

    print()

    # Now try to import vector store
    print("Importing vector store module...")
    try:
        from src.vector_store import VectorStoreManager
        print("✓ Vector store module imported")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return

    print()

    # Create test documents
    test_docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and AI.",
            metadata={"source": "test", "topic": "python", "document_id": 1, "chunk_index": 0}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
            metadata={"source": "test", "topic": "ml", "document_id": 1, "chunk_index": 1}
        ),
        Document(
            page_content="ChromaDB is an open-source embedding database designed for AI applications. It provides fast similarity search and easy integration.",
            metadata={"source": "test", "topic": "database", "document_id": 2, "chunk_index": 0}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. It helps build RAG systems and AI agents.",
            metadata={"source": "test", "topic": "ai", "document_id": 2, "chunk_index": 1}
        ),
    ]

    print("Test documents created:")
    for i, doc in enumerate(test_docs, 1):
        print(f"  {i}. {doc.page_content[:60]}...")
    print()

    # Initialize vector store
    print("Initializing vector store (this may take a moment)...")
    try:
        vs = VectorStoreManager(project_id=999)
        print("✓ Vector store initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize vector store: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Clear any existing data
    print("Clearing any existing test data...")
    clear_result = vs.clear_all()
    print(f"  {clear_result.get('message', 'Cleared')}")
    print()

    # Get initial stats
    print("Initial Statistics:")
    stats = vs.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Add documents
    print("Adding test documents to vector store...")
    print("(This will generate embeddings using Ollama - may take 10-30 seconds)")
    try:
        result = vs.add_documents(test_docs)
        if result["success"]:
            print(f"✓ {result['message']}")
        else:
            print(f"✗ Error: {result['error']}")
            return
    except Exception as e:
        print(f"✗ Failed to add documents: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Get updated stats
    print("Updated Statistics:")
    stats = vs.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Test 1: Similarity search
    print("Test 1: Similarity Search")
    print("-" * 60)
    queries = [
        "What is Python?",
        "Tell me about databases",
        "Explain machine learning"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = vs.similarity_search(query, k=2)
        print(f"Found {len(results)} results:")

        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Content: {doc.page_content[:80]}...")
            print(f"  Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"  Document ID: {doc.metadata.get('document_id', 'N/A')}")

    print()

    # Test 2: Search with scores
    print("\nTest 2: Similarity Search with Scores")
    print("-" * 60)
    query = "programming languages"
    print(f"Query: '{query}'")
    results_with_scores = vs.similarity_search_with_score(query, k=3)
    print(f"Found {len(results_with_scores)} results:\n")

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"  Result {i} (relevance score: {score:.4f}):")
        print(f"  Content: {doc.page_content[:80]}...")
        print(f"  Topic: {doc.metadata.get('topic', 'N/A')}")
        print()

    # Test 3: Metadata search
    print("Test 3: Search by Metadata")
    print("-" * 60)
    print("Searching for documents with document_id=1...")
    docs_by_id = vs.search_by_metadata({"document_id": 1}, limit=10)
    print(f"Found {len(docs_by_id)} chunks")
    for doc in docs_by_id:
        print(f"  - Chunk {doc.metadata.get('chunk_index')}: {doc.page_content[:60]}...")
    print()

    # Test 4: Get all document IDs
    print("Test 4: Get All Document IDs")
    print("-" * 60)
    doc_ids = vs.get_all_document_ids()
    print(f"Document IDs in vector store: {doc_ids}")
    print()

    # Test 5: Delete by document ID
    print("Test 5: Delete Documents by ID")
    print("-" * 60)
    print("Deleting document_id=1...")
    delete_result = vs.delete_by_document_id(1)
    if delete_result["success"]:
        print(f"✓ {delete_result['message']}")
        print(f"  Deleted {delete_result['deleted_count']} chunks")
    else:
        print(f"✗ {delete_result['error']}")
    print()

    # Check stats after deletion
    stats = vs.get_stats()
    print(f"Remaining chunks: {stats['total_chunks']}")
    doc_ids = vs.get_all_document_ids()
    print(f"Remaining document IDs: {doc_ids}")
    print()

    # Clean up
    print("Cleaning up test data...")
    clear_result = vs.clear_all()
    print(f"✓ {clear_result['message']}")

    print()
    print("=" * 60)
    print("Vector Store Tests Complete! ✓")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
