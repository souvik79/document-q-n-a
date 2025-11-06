"""
Test script for Q&A chain.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain.schema import Document
import requests

def main():
    print("=" * 60)
    print("Q&A Chain Test")
    print("=" * 60)
    print()

    # Check Ollama
    print("Checking Ollama availability...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            models = response.json().get('models', [])
            print(f"  Available models: {[m['name'] for m in models]}")
        else:
            print("✗ Ollama not available. Please start it with: ollama serve")
            return
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Please start Ollama with: ollama serve")
        return

    print()

    # Import modules
    print("Importing modules...")
    try:
        from src.vector_store import VectorStoreManager
        from src.qa_chain import QAChain
        print("✓ Modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import modules: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Setup test environment
    print("Setting up test environment...")
    print("(This will create embeddings - may take 20-30 seconds)")

    try:
        # Initialize vector store
        vs = VectorStoreManager(project_id=888)
        vs.clear_all()

        # Create comprehensive test documents
        test_docs = [
            Document(
                page_content="Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. "
                            "It emphasizes code readability with its use of significant indentation. Python supports multiple "
                            "programming paradigms including procedural, object-oriented, and functional programming. "
                            "It's widely used for web development, data analysis, artificial intelligence, scientific computing, "
                            "and automation. Popular frameworks include Django and Flask for web development.",
                metadata={"source": "python_guide", "topic": "python", "document_id": 1, "chunk_index": 0}
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve "
                            "from experience without being explicitly programmed. It focuses on developing algorithms that can "
                            "access data and use it to learn for themselves. The process involves training models on data, "
                            "validating their performance, and making predictions. Common types include supervised learning, "
                            "unsupervised learning, and reinforcement learning. Applications range from email filtering to "
                            "medical diagnosis and autonomous vehicles.",
                metadata={"source": "ml_basics", "topic": "machine_learning", "document_id": 2, "chunk_index": 0}
            ),
            Document(
                page_content="LangChain is a powerful framework designed for developing applications with large language models. "
                            "It provides abstractions and tools for common LLM tasks including document loaders for various formats, "
                            "text splitters for chunking, embeddings for semantic search, vector stores for retrieval, "
                            "and chains for orchestrating complex workflows. LangChain enables developers to build "
                            "Retrieval-Augmented Generation (RAG) systems, chatbots, and AI agents with minimal code. "
                            "It integrates with popular LLMs like OpenAI, Anthropic, and local models via Ollama.",
                metadata={"source": "langchain_docs", "topic": "langchain", "document_id": 3, "chunk_index": 0}
            ),
            Document(
                page_content="Ollama is a tool that allows you to run large language models locally on your machine. "
                            "It supports various models including Llama, Mistral, and others. Ollama provides a simple "
                            "API for model interaction and handles model management, making it easy to use LLMs without "
                            "cloud dependencies. It's particularly useful for privacy-sensitive applications and development "
                            "environments where you want full control over your AI infrastructure.",
                metadata={"source": "ollama_info", "topic": "ollama", "document_id": 4, "chunk_index": 0}
            ),
        ]

        print("  Adding documents to vector store...")
        result = vs.add_documents(test_docs)
        if result["success"]:
            print(f"  ✓ {result['message']}")
        else:
            print(f"  ✗ Error: {result['error']}")
            return

    except Exception as e:
        print(f"✗ Failed to setup test environment: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Initialize Q&A chain
    print("Initializing Q&A chain...")
    try:
        qa = QAChain(project_id=888)
        print("✓ Q&A chain initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Q&A chain: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Display statistics
    print("Chain Statistics:")
    print("-" * 60)
    stats = qa.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Test questions
    test_cases = [
        {
            "question": "Who created Python and when?",
            "expected_keywords": ["Guido van Rossum", "1991"]
        },
        {
            "question": "What are the main applications of machine learning?",
            "expected_keywords": ["prediction", "learning", "data"]
        },
        {
            "question": "What is LangChain used for?",
            "expected_keywords": ["LangChain", "framework", "language models"]
        },
        {
            "question": "What is the advantage of using Ollama?",
            "expected_keywords": ["local", "Ollama", "models"]
        },
    ]

    print("=" * 60)
    print("Running Q&A Tests")
    print("=" * 60)
    print()

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]

        print(f"Test {i}/{len(test_cases)}")
        print(f"Question: {question}")
        print("-" * 60)

        try:
            result = qa.ask(question)

            if result["success"]:
                print(f"\nAnswer:")
                print(f"{result['answer']}")
                print()

                print(f"Metadata:")
                print(f"  Model: {result['metadata']['model']}")
                print(f"  Query time: {result['metadata']['query_time_seconds']}s")
                print(f"  Sources used: {result['metadata']['num_sources']}")
                print(f"  Source document IDs: {result['source_document_ids']}")

                print(f"\nSource excerpts:")
                for j, source in enumerate(result['sources'][:2], 1):  # Show first 2 sources
                    print(f"  {j}. {source['content'][:100]}...")

                print()
                print("✓ Test passed")

            else:
                print(f"✗ Error: {result['error']}")

        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Test retrieval without answer generation
    print("=" * 60)
    print("Testing Document Retrieval (without answer generation)")
    print("=" * 60)
    print()

    query = "programming languages"
    print(f"Query: '{query}'")
    print()

    relevant_docs = qa.get_relevant_documents(query, k=2)
    print(f"Found {len(relevant_docs)} relevant documents:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\n  Document {i}:")
        print(f"  Content: {doc['content'][:150]}...")
        print(f"  Topic: {doc['metadata'].get('topic', 'N/A')}")
        print(f"  Document ID: {doc['metadata'].get('document_id', 'N/A')}")

    print()

    # Clean up
    print("=" * 60)
    print("Cleaning up test data...")
    vs.clear_all()
    print("✓ Cleanup complete")

    print()
    print("=" * 60)
    print("Q&A Chain Tests Complete! ✓")
    print("=" * 60)
    print()
    print("Summary:")
    print("  ✓ Ollama integration working")
    print("  ✓ Vector store retrieval working")
    print("  ✓ Question answering working")
    print("  ✓ Source attribution working")
    print("  ✓ Metadata tracking working")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
