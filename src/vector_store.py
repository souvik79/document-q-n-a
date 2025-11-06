"""
Vector store management using ChromaDB.
Handles document embeddings, storage, and similarity search.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    TOP_K_RESULTS,
    SEARCH_TYPE,
    get_chroma_dir,
)


class VectorStoreManager:
    """Manages vector database operations for document embeddings and retrieval."""

    def __init__(self, project_id: int, embedding_model: Optional[str] = None):
        """
        Initialize vector store manager.

        Args:
            project_id: Project ID for isolating vector stores
            embedding_model: Optional embedding model override
        """
        self.project_id = project_id
        self.persist_directory = str(get_chroma_dir(project_id))

        # Initialize embeddings
        self.embedding_model = embedding_model or OLLAMA_EMBEDDING_MODEL
        self.embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=self.embedding_model,
        )

        # Initialize vector store (will load if exists, create if not)
        self.vectorstore = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize or load the ChromaDB vector store."""
        try:
            # Try to load existing vectorstore
            if Path(self.persist_directory).exists():
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=f"project_{self.project_id}",
                )
            else:
                # Create new vectorstore
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=f"project_{self.project_id}",
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

    def add_documents(self, documents: List[Document], document_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects
            document_id: Optional document ID to associate with chunks

        Returns:
            Dictionary with operation result
        """
        try:
            if not documents:
                return {"success": False, "error": "No documents provided"}

            # Add document_id to metadata if provided
            if document_id is not None:
                for doc in documents:
                    doc.metadata["document_id"] = document_id

            # Generate unique IDs for each document chunk
            ids = [str(uuid.uuid4()) for _ in documents]

            # Add to vector store
            self.vectorstore.add_documents(documents=documents, ids=ids)

            # Persist to disk
            self.vectorstore.persist()

            return {
                "success": True,
                "added_count": len(documents),
                "ids": ids,
                "message": f"Successfully added {len(documents)} document chunks"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to add documents: {str(e)}"
            }

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Query text
            k: Number of results to return (default: TOP_K_RESULTS)
            filter_metadata: Optional metadata filter

        Returns:
            List of similar Document objects
        """
        try:
            k = k or TOP_K_RESULTS

            if filter_metadata:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )

            return results

        except Exception as e:
            print(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        try:
            k = k or TOP_K_RESULTS

            if filter_metadata:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )

            return results

        except Exception as e:
            print(f"Similarity search with score failed: {e}")
            return []

    def mmr_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Perform Maximal Marginal Relevance (MMR) search.
        MMR balances relevance and diversity in results.

        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of candidates to consider (default: 4*k)
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance)

        Returns:
            List of diverse, relevant Document objects
        """
        try:
            k = k or TOP_K_RESULTS
            fetch_k = fetch_k or (k * 4)

            results = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )

            return results

        except Exception as e:
            print(f"MMR search failed: {e}")
            return []

    def get_retriever(self, search_type: Optional[str] = None, k: Optional[int] = None):
        """
        Get a LangChain retriever for use in chains.

        Args:
            search_type: Type of search ("similarity" or "mmr")
            k: Number of results

        Returns:
            LangChain retriever object
        """
        search_type = search_type or SEARCH_TYPE
        k = k or TOP_K_RESULTS

        if search_type == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 4}
            )
        else:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )

    def delete_by_document_id(self, document_id: int) -> Dict[str, Any]:
        """
        Delete all chunks associated with a document ID.

        Args:
            document_id: Document ID to delete

        Returns:
            Dictionary with operation result
        """
        try:
            # Get all documents with this document_id
            collection = self.vectorstore._collection
            results = collection.get(
                where={"document_id": document_id}
            )

            if results and results['ids']:
                # Delete the documents
                collection.delete(ids=results['ids'])

                # Persist changes
                self.vectorstore.persist()

                return {
                    "success": True,
                    "deleted_count": len(results['ids']),
                    "message": f"Deleted {len(results['ids'])} chunks"
                }
            else:
                return {
                    "success": True,
                    "deleted_count": 0,
                    "message": "No chunks found for this document"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete documents: {str(e)}"
            }

    def clear_all(self) -> Dict[str, Any]:
        """
        Clear all documents from the vector store.

        Returns:
            Dictionary with operation result
        """
        try:
            # Get collection
            collection = self.vectorstore._collection

            # Get all IDs
            all_data = collection.get()

            if all_data and all_data['ids']:
                # Delete all
                collection.delete(ids=all_data['ids'])

                # Persist changes
                self.vectorstore.persist()

                return {
                    "success": True,
                    "deleted_count": len(all_data['ids']),
                    "message": f"Cleared {len(all_data['ids'])} document chunks"
                }
            else:
                return {
                    "success": True,
                    "deleted_count": 0,
                    "message": "Vector store is already empty"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to clear vector store: {str(e)}"
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "total_chunks": count,
                "collection_name": collection.name,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model,
            }

        except Exception as e:
            return {
                "error": f"Failed to get stats: {str(e)}"
            }

    def compact_database(self) -> Dict[str, Any]:
        """
        Compact/optimize the vector database to reclaim disk space after deletions.

        This recreates the collection with only active documents, removing deleted data.

        Returns:
            Dictionary with compaction results
        """
        try:
            import shutil
            from pathlib import Path

            # Get all current documents
            collection = self.vectorstore._collection
            all_data = collection.get()

            if not all_data or not all_data['ids']:
                return {
                    "success": True,
                    "message": "Collection is empty, nothing to compact",
                    "space_saved_mb": 0
                }

            # Get size before
            size_before = sum(f.stat().st_size for f in Path(self.persist_directory).rglob('*') if f.is_file())

            # Backup directory name
            backup_dir = Path(self.persist_directory).parent / f"{Path(self.persist_directory).name}_backup"

            # Create backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(self.persist_directory, backup_dir)

            try:
                # Delete the old collection
                collection_name = collection.name
                self.client.delete_collection(collection_name)

                # Recreate collection
                new_collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )

                # Re-add all documents
                if all_data['ids']:
                    new_collection.add(
                        ids=all_data['ids'],
                        embeddings=all_data['embeddings'],
                        documents=all_data['documents'],
                        metadatas=all_data['metadatas']
                    )

                # Reinitialize vectorstore
                self.vectorstore = Chroma(
                    client=self.client,
                    collection_name=collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )

                # Persist
                self.vectorstore.persist()

                # Remove backup
                shutil.rmtree(backup_dir)

                # Get size after
                size_after = sum(f.stat().st_size for f in Path(self.persist_directory).rglob('*') if f.is_file())
                space_saved = size_before - size_after

                return {
                    "success": True,
                    "message": "Database compacted successfully",
                    "chunks_kept": len(all_data['ids']),
                    "size_before_mb": round(size_before / (1024 * 1024), 2),
                    "size_after_mb": round(size_after / (1024 * 1024), 2),
                    "space_saved_mb": round(space_saved / (1024 * 1024), 2)
                }

            except Exception as e:
                # Restore from backup on error
                if Path(self.persist_directory).exists():
                    shutil.rmtree(self.persist_directory)
                shutil.copytree(backup_dir, self.persist_directory)
                shutil.rmtree(backup_dir)
                raise e

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to compact database: {str(e)}"
            }

    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 100) -> List[Document]:
        """
        Search documents by metadata filter.

        Args:
            metadata_filter: Dictionary of metadata key-value pairs
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        try:
            collection = self.vectorstore._collection
            results = collection.get(
                where=metadata_filter,
                limit=limit
            )

            documents = []
            if results and results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    doc = Document(
                        page_content=results['documents'][i] if results['documents'] else "",
                        metadata=results['metadatas'][i] if results['metadatas'] else {}
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            print(f"Metadata search failed: {e}")
            return []

    def get_all_document_ids(self) -> List[int]:
        """
        Get all unique document IDs in the vector store.

        Returns:
            List of document IDs
        """
        try:
            collection = self.vectorstore._collection
            results = collection.get()

            document_ids = set()
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    if 'document_id' in metadata:
                        document_ids.add(metadata['document_id'])

            return sorted(list(document_ids))

        except Exception as e:
            print(f"Failed to get document IDs: {e}")
            return []


def create_vector_store(project_id: int) -> VectorStoreManager:
    """
    Create a new vector store manager for a project.

    Args:
        project_id: Project ID

    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(project_id)


def get_vector_store(project_id: int) -> VectorStoreManager:
    """
    Get or create a vector store manager for a project.

    Args:
        project_id: Project ID

    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(project_id)


if __name__ == "__main__":
    # Test vector store
    print("=" * 60)
    print("Vector Store Test")
    print("=" * 60)
    print()

    # Create test documents
    test_docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"source": "test", "topic": "python", "document_id": 1}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            metadata={"source": "test", "topic": "ml", "document_id": 1}
        ),
        Document(
            page_content="ChromaDB is an open-source embedding database designed for AI applications.",
            metadata={"source": "test", "topic": "database", "document_id": 2}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "test", "topic": "ai", "document_id": 2}
        ),
    ]

    # Initialize vector store (use test project ID)
    print("Initializing vector store...")
    try:
        vs = VectorStoreManager(project_id=999)
        print("✓ Vector store initialized")
        print()

        # Get initial stats
        stats = vs.get_stats()
        print(f"Initial stats: {stats}")
        print()

        # Add documents
        print("Adding test documents...")
        result = vs.add_documents(test_docs)
        if result["success"]:
            print(f"✓ {result['message']}")
        else:
            print(f"✗ Error: {result['error']}")
        print()

        # Get updated stats
        stats = vs.get_stats()
        print(f"Updated stats: {stats}")
        print()

        # Test similarity search
        print("Testing similarity search...")
        query = "What is Python?"
        results = vs.similarity_search(query, k=2)
        print(f"Query: '{query}'")
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")

        print()

        # Test search with scores
        print("Testing similarity search with scores...")
        query = "Tell me about databases"
        results_with_scores = vs.similarity_search_with_score(query, k=2)
        print(f"Query: '{query}'")
        print(f"Found {len(results_with_scores)} results:")
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n  Result {i} (score: {score:.4f}):")
            print(f"  Content: {doc.page_content[:100]}...")

        print()

        # Test metadata search
        print("Testing metadata search...")
        docs_by_id = vs.search_by_metadata({"document_id": 1}, limit=10)
        print(f"Documents with document_id=1: {len(docs_by_id)}")

        print()

        # Get all document IDs
        print("Getting all document IDs...")
        doc_ids = vs.get_all_document_ids()
        print(f"Document IDs in vector store: {doc_ids}")

        print()

        # Clean up
        print("Cleaning up test data...")
        clear_result = vs.clear_all()
        if clear_result["success"]:
            print(f"✓ {clear_result['message']}")

        print()
        print("=" * 60)
        print("Vector Store Tests Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
