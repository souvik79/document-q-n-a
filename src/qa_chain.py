"""
Q&A Chain module for question answering using RAG (Retrieval-Augmented Generation).
Combines vector store retrieval with Ollama LLM for accurate, context-based answers.
"""

from typing import Dict, Any, List, Optional
import time

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .vector_store import VectorStoreManager
from .improved_retrieval import ImprovedRetriever
from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K_RESULTS,
    LLM_PROVIDER,
    GOOGLE_API_KEY,
)


# Default prompt template for Q&A
DEFAULT_QA_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context.

Context information is below:
---------------------
{context}
---------------------

Using the context information above, please answer the following question.
If you cannot find the answer in the context, say "I don't have enough information in the provided documents to answer this question."
Always cite which part of the context you used to form your answer.

Question: {question}

Answer: """


class QAChain:
    """Question-Answering chain using RAG pattern."""

    def __init__(
        self,
        project_id: int,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        streaming: bool = False,
        use_improved_retrieval: bool = True,
        provider: Optional[str] = None,
    ):
        """
        Initialize Q&A chain.

        Args:
            project_id: Project ID for vector store
            model: Model name (default from config)
            temperature: LLM temperature (default from config)
            streaming: Enable streaming output
            use_improved_retrieval: Use hybrid search with keyword matching (default True)
            provider: 'local' or 'gemini' (default from config)
        """
        self.project_id = project_id
        self.model_name = model or OLLAMA_MODEL
        self.temperature = temperature if temperature is not None else TEMPERATURE
        self.use_improved_retrieval = use_improved_retrieval
        self.provider = provider or LLM_PROVIDER

        # Initialize vector store
        self.vector_store = VectorStoreManager(project_id)

        # Initialize improved retriever if enabled
        if use_improved_retrieval:
            self.improved_retriever = ImprovedRetriever(self.vector_store)

        # Initialize LLM based on provider
        if self.provider == "gemini":
            # Use Gemini API
            if not GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY not set. Please set it in .env file. "
                    "Get your free key from: https://makersuite.google.com/app/apikey"
                )
            from .cloud_providers import get_gemini_provider
            self.gemini_provider = get_gemini_provider()
            self.llm = None  # Not using LangChain LLM for Gemini
        else:
            # Use local Ollama
            callback_manager = None
            if streaming:
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            self.llm = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=self.model_name,
                temperature=self.temperature,
                num_predict=MAX_TOKENS,
                callback_manager=callback_manager,
            )
            self.gemini_provider = None

        # Create prompt template
        self.prompt_template = PromptTemplate(
            template=DEFAULT_QA_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Initialize retrieval chain (only for local Ollama)
        if self.provider == "local":
            self._initialize_chain()

    def _initialize_chain(self):
        """Initialize the retrieval Q&A chain."""
        # Get retriever from vector store
        retriever = self.vector_store.get_retriever(k=TOP_K_RESULTS)

        # Create RetrievalQA chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" puts all retrieved docs into prompt
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.

        Args:
            question: User question

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                "success": False,
                "error": "Question cannot be empty"
            }

        try:
            # Check if vector store has documents
            stats = self.vector_store.get_stats()
            if stats.get("total_chunks", 0) == 0:
                return {
                    "success": False,
                    "error": "No documents have been added to this project yet. Please add documents first."
                }

            # Time the query
            start_time = time.time()

            # Use improved retrieval if enabled
            if self.use_improved_retrieval:
                # Get relevant documents using improved retrieval
                source_documents = self.improved_retriever.smart_retrieval_for_api_docs(
                    question, k=TOP_K_RESULTS
                )

                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in source_documents])

                # Format prompt with context
                formatted_prompt = self.prompt_template.format(
                    context=context,
                    question=question
                )

                # Get answer from LLM based on provider
                if self.provider == "gemini":
                    # Use Gemini API
                    answer = self.gemini_provider.generate_text(
                        formatted_prompt,
                        temperature=self.temperature
                    )
                else:
                    # Use local Ollama
                    answer = self.llm.invoke(formatted_prompt)

                query_time = time.time() - start_time
            else:
                # Use standard retrieval chain (only for local)
                if self.provider == "gemini":
                    # For Gemini, we must use improved retrieval
                    raise ValueError("Standard retrieval not supported with Gemini. Set use_improved_retrieval=True")

                result = self.chain({"query": question})
                query_time = time.time() - start_time

                # Extract answer and sources
                answer = result.get("result", "")
                source_documents = result.get("source_documents", [])

            # Process source documents
            sources = []
            source_document_ids = set()

            for doc in source_documents:
                source_info = {
                    "content": doc.page_content,  # Store full content for source viewer
                    "metadata": doc.metadata,
                }

                # Track document IDs
                if "document_id" in doc.metadata:
                    source_document_ids.add(doc.metadata["document_id"])

                sources.append(source_info)

            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_document_ids": list(source_document_ids),
                "metadata": {
                    "provider": self.provider,
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "query_time_seconds": round(query_time, 2),
                    "num_sources": len(sources),
                    "retrieval_method": "improved_hybrid" if self.use_improved_retrieval else "standard",
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate answer: {str(e)}"
            }

    def ask_with_context(self, question: str, document_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Ask a question with optional filtering by document IDs.

        Args:
            question: User question
            document_ids: Optional list of document IDs to search within

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not question or not question.strip():
            return {
                "success": False,
                "error": "Question cannot be empty"
            }

        try:
            # If document_ids provided, do filtered search
            if document_ids:
                # Get relevant documents with filtering
                relevant_docs = []
                for doc_id in document_ids:
                    docs = self.vector_store.similarity_search(
                        question,
                        k=TOP_K_RESULTS,
                        filter_metadata={"document_id": doc_id}
                    )
                    relevant_docs.extend(docs)

                if not relevant_docs:
                    return {
                        "success": False,
                        "error": "No relevant content found in the specified documents"
                    }

                # Create context from relevant docs
                context = "\n\n".join([doc.page_content for doc in relevant_docs[:TOP_K_RESULTS]])

                # Format prompt with context
                formatted_prompt = self.prompt_template.format(
                    context=context,
                    question=question
                )

                # Time the query
                start_time = time.time()

                # Get answer from LLM (using invoke instead of deprecated __call__)
                answer = self.llm.invoke(formatted_prompt)

                query_time = time.time() - start_time

                # Process sources
                sources = []
                for doc in relevant_docs[:TOP_K_RESULTS]:
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata,
                    }
                    sources.append(source_info)

                return {
                    "success": True,
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "source_document_ids": document_ids,
                    "metadata": {
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "query_time_seconds": round(query_time, 2),
                        "num_sources": len(sources),
                        "filtered": True,
                    }
                }

            else:
                # Use regular ask method
                return self.ask(question)

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate answer: {str(e)}"
            }

    def get_relevant_documents(self, question: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get relevant document chunks for a question without generating an answer.

        Args:
            question: User question
            k: Number of documents to retrieve

        Returns:
            List of relevant document chunks
        """
        try:
            k = k or TOP_K_RESULTS
            docs = self.vector_store.similarity_search(question, k=k)

            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })

            return results

        except Exception as e:
            print(f"Failed to retrieve documents: {e}")
            return []

    def update_prompt(self, template: str):
        """
        Update the prompt template.

        Args:
            template: New prompt template string (must include {context} and {question})
        """
        if "{context}" not in template or "{question}" not in template:
            raise ValueError("Template must include {context} and {question} placeholders")

        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Reinitialize chain with new prompt
        self._initialize_chain()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Q&A chain.

        Returns:
            Dictionary with statistics
        """
        vector_stats = self.vector_store.get_stats()

        return {
            "project_id": self.project_id,
            "model": self.model_name,
            "temperature": self.temperature,
            "total_chunks": vector_stats.get("total_chunks", 0),
            "embedding_model": vector_stats.get("embedding_model", "unknown"),
            "top_k_results": TOP_K_RESULTS,
        }


def create_qa_chain(project_id: int, **kwargs) -> QAChain:
    """
    Factory function to create a Q&A chain.

    Args:
        project_id: Project ID
        **kwargs: Additional arguments for QAChain

    Returns:
        QAChain instance
    """
    return QAChain(project_id, **kwargs)


if __name__ == "__main__":
    # Test Q&A chain
    import sys
    from pathlib import Path

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from langchain.schema import Document

    print("=" * 60)
    print("Q&A Chain Test")
    print("=" * 60)
    print()

    # Check Ollama
    print("Checking Ollama...")
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
        else:
            print("✗ Ollama not available")
            sys.exit(1)
    except:
        print("✗ Cannot connect to Ollama")
        sys.exit(1)

    print()

    # Create test documents
    print("Setting up test environment...")
    from src.vector_store import VectorStoreManager

    # Initialize vector store
    vs = VectorStoreManager(project_id=888)
    vs.clear_all()

    # Add test documents
    test_docs = [
        Document(
            page_content="Python is a high-level, interpreted programming language created by Guido van Rossum. "
                        "It was first released in 1991 and is known for its simple, readable syntax. "
                        "Python is widely used in web development, data science, artificial intelligence, and automation.",
            metadata={"source": "test", "topic": "python", "document_id": 1}
        ),
        Document(
            page_content="Machine learning is a branch of artificial intelligence that enables computers to learn "
                        "from data without being explicitly programmed. It uses algorithms to identify patterns "
                        "and make predictions. Common applications include image recognition, natural language processing, "
                        "and recommendation systems.",
            metadata={"source": "test", "topic": "ml", "document_id": 2}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. "
                        "It provides tools for document loading, text splitting, embeddings, vector stores, "
                        "and chains for building RAG (Retrieval-Augmented Generation) systems. "
                        "LangChain simplifies the process of creating AI applications.",
            metadata={"source": "test", "topic": "langchain", "document_id": 3}
        ),
    ]

    print("Adding test documents to vector store...")
    vs.add_documents(test_docs)
    print("✓ Documents added")
    print()

    # Initialize Q&A chain
    print("Initializing Q&A chain...")
    qa = QAChain(project_id=888)
    print("✓ Q&A chain initialized")
    print()

    # Get stats
    stats = qa.get_stats()
    print("Chain Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Test questions
    questions = [
        "Who created Python?",
        "What is machine learning used for?",
        "What is LangChain?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 60)

        result = qa.ask(question)

        if result["success"]:
            print(f"Answer: {result['answer']}\n")
            print(f"Query time: {result['metadata']['query_time_seconds']}s")
            print(f"Sources used: {result['metadata']['num_sources']}")
            print(f"Source document IDs: {result['source_document_ids']}")
        else:
            print(f"Error: {result['error']}")

        print()

    # Clean up
    print("Cleaning up...")
    vs.clear_all()
    print("✓ Test complete")

    print()
    print("=" * 60)
    print("Q&A Chain Test Complete!")
    print("=" * 60)
