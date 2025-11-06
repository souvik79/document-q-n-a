#!/usr/bin/env python3
"""
Debug tool for testing and improving retrieval quality.

This script helps diagnose why certain chunks are being retrieved
and provides suggestions for improving retrieval accuracy.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.session_manager import SessionManager
from src.vector_store import VectorStoreManager
from src.qa_chain import QAChain
from src.improved_retrieval import ImprovedRetriever


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")


def print_document(index, doc, score=None):
    """Print document details."""
    print(f"\n--- Document {index} {'(Score: ' + str(round(score, 3)) + ')' if score else ''} ---")
    print(f"Content (first 300 chars):\n{doc.page_content[:300]}...")
    print(f"\nMetadata: {doc.metadata}")
    print("-" * 80)


def test_retrieval(project_id: int, question: str, k: int = 10):
    """
    Test retrieval quality for a given question.

    Args:
        project_id: Project ID to test
        question: Test question
        k: Number of results to retrieve
    """
    print_header(f"Testing Retrieval for Project {project_id}")

    print(f"Question: {question}\n")

    # Initialize components
    print("Initializing components...")
    vsm = VectorStoreManager(project_id)
    improved_retriever = ImprovedRetriever(vsm)

    # Check stats
    stats = vsm.get_stats()
    print(f"✓ Vector store has {stats.get('total_chunks', 0)} chunks\n")

    if stats.get('total_chunks', 0) == 0:
        print("❌ No documents in project. Add documents first!")
        return

    # Test 1: Standard similarity search
    print_header("Test 1: Standard Similarity Search (Current Method)")
    try:
        standard_results = vsm.search(question, k=k)
        print(f"Retrieved {len(standard_results)} documents:\n")

        for i, doc in enumerate(standard_results[:5], 1):  # Show top 5
            print_document(i, doc)

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Hybrid search
    print_header("Test 2: Hybrid Search (Semantic + Keywords)")
    try:
        hybrid_results = improved_retriever.hybrid_search(question, k=5)
        print(f"Retrieved {len(hybrid_results)} documents:\n")

        for i, doc in enumerate(hybrid_results, 1):
            print_document(i, doc)

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: API-optimized search
    print_header("Test 3: API-Optimized Search")
    try:
        api_results = improved_retriever.smart_retrieval_for_api_docs(question, k=5)
        print(f"Retrieved {len(api_results)} documents:\n")

        for i, doc in enumerate(api_results, 1):
            print_document(i, doc)

    except Exception as e:
        print(f"❌ Error: {e}")

    # Analysis
    print_header("Analysis & Recommendations")

    # Extract keywords
    keywords = improved_retriever._extract_keywords(question)
    print(f"Extracted Keywords: {keywords}\n")

    # Check if API-related
    api_keywords = ['api', 'post', 'get', 'put', 'delete', 'endpoint', 'request', 'call']
    is_api_query = any(kw in question.lower() for kw in api_keywords)
    print(f"API-related query: {'Yes' if is_api_query else 'No'}\n")

    # Recommendations
    print("Recommendations:")
    print("1. If results are poor, try:")
    print("   - Smaller chunk size (500-800 chars) for better granularity")
    print("   - Larger chunk overlap (250-300 chars) for better context")
    print("   - Retrieve more chunks (k=10-15) for better coverage")
    print("\n2. For API documentation:")
    print("   - Ensure POST/GET/endpoint descriptions are in same chunk")
    print("   - Add metadata tags (api_method, endpoint_path) during ingestion")
    print("   - Use hybrid search (equal semantic + keyword weight)")
    print("\n3. Update your .env file:")
    print("   CHUNK_SIZE=700")
    print("   CHUNK_OVERLAP=250")
    print("   TOP_K_RESULTS=10")


def compare_retrievals(project_id: int, question: str):
    """
    Compare different retrieval methods side-by-side.

    Args:
        project_id: Project ID
        question: Test question
    """
    print_header("Retrieval Method Comparison")

    vsm = VectorStoreManager(project_id)
    improved_retriever = ImprovedRetriever(vsm)

    methods = {
        "Standard (Current)": lambda: vsm.search(question, k=3),
        "Hybrid (Improved)": lambda: improved_retriever.hybrid_search(question, k=3),
        "API-Optimized": lambda: improved_retriever.smart_retrieval_for_api_docs(question, k=3),
    }

    for method_name, method_func in methods.items():
        print(f"\n{method_name}:")
        print("-" * 80)
        try:
            results = method_func()
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.page_content[:150]}...")
        except Exception as e:
            print(f"Error: {e}")


def show_all_chunks(project_id: int, max_chunks: int = 20):
    """
    Show all chunks in the vector store for inspection.

    Args:
        project_id: Project ID
        max_chunks: Maximum chunks to show
    """
    print_header(f"All Chunks in Project {project_id}")

    vsm = VectorStoreManager(project_id)

    # Get stats
    stats = vsm.get_stats()
    total_chunks = stats.get('total_chunks', 0)

    print(f"Total chunks: {total_chunks}")
    print(f"Showing first {min(max_chunks, total_chunks)} chunks:\n")

    # Get all documents (using a generic query)
    all_docs = vsm.search("document", k=max_chunks)

    for i, doc in enumerate(all_docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Content:\n{doc.page_content[:200]}...")
        print(f"\nMetadata: {doc.metadata}")
        print("-" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Debug Retrieval Tool")
        print("\nUsage:")
        print("  python debug_retrieval.py test <project_id> <question>")
        print("  python debug_retrieval.py compare <project_id> <question>")
        print("  python debug_retrieval.py show-chunks <project_id>")
        print("\nExamples:")
        print("  python debug_retrieval.py test 2 \"How to do a POST call?\"")
        print("  python debug_retrieval.py compare 2 \"API endpoint for listing\"")
        print("  python debug_retrieval.py show-chunks 2")
        return

    command = sys.argv[1]

    if command == "test":
        if len(sys.argv) < 4:
            print("❌ Usage: python debug_retrieval.py test <project_id> <question>")
            return

        project_id = int(sys.argv[2])
        question = " ".join(sys.argv[3:])

        test_retrieval(project_id, question)

    elif command == "compare":
        if len(sys.argv) < 4:
            print("❌ Usage: python debug_retrieval.py compare <project_id> <question>")
            return

        project_id = int(sys.argv[2])
        question = " ".join(sys.argv[3:])

        compare_retrievals(project_id, question)

    elif command == "show-chunks":
        if len(sys.argv) < 3:
            print("❌ Usage: python debug_retrieval.py show-chunks <project_id>")
            return

        project_id = int(sys.argv[2])
        max_chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 20

        show_all_chunks(project_id, max_chunks)

    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: test, compare, show-chunks")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
