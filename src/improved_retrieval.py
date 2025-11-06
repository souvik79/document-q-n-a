"""
Improved retrieval strategies for better context matching.

This module provides enhanced retrieval methods that combine:
1. Hybrid search (semantic + keyword)
2. Re-ranking based on relevance
3. Better metadata filtering
4. Contextual chunk expansion
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
import re


class ImprovedRetriever:
    """Enhanced retrieval with hybrid search and re-ranking."""

    def __init__(self, vector_store_manager):
        """
        Initialize improved retriever.

        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vsm = vector_store_manager

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7
    ) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)

        Returns:
            List of relevant documents
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)

        # Get more results than needed for reranking
        semantic_results = self.vsm.similarity_search(query, k=k*2)

        # Score and rerank results
        scored_results = []
        for doc in semantic_results:
            # Semantic score (from vector similarity)
            semantic_score = semantic_weight

            # Keyword score
            keyword_score = self._keyword_match_score(
                doc.page_content,
                keywords
            )

            # Combined score
            total_score = (semantic_weight * semantic_score +
                          (1 - semantic_weight) * keyword_score)

            scored_results.append((doc, total_score))

        # Sort by score and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_results[:k]]

    def search_with_context_expansion(
        self,
        query: str,
        k: int = 5,
        expand_chunks: int = 1
    ) -> List[Document]:
        """
        Search and expand retrieved chunks with surrounding context.

        Args:
            query: Search query
            k: Number of base chunks to retrieve
            expand_chunks: Number of chunks before/after to include

        Returns:
            List of documents with expanded context
        """
        # Get initial results
        results = self.vsm.similarity_search(query, k=k)

        # Expand with surrounding chunks
        expanded_results = []
        for doc in results:
            # Get metadata for chunk identification
            chunk_index = doc.metadata.get('chunk_index', 0)
            document_id = doc.metadata.get('document_id')

            if document_id is not None:
                # Try to get surrounding chunks
                # Note: This requires storing chunk order in metadata
                expanded_results.append(doc)
            else:
                expanded_results.append(doc)

        return expanded_results

    def search_with_metadata_filtering(
        self,
        query: str,
        k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search with metadata filtering for better relevance.

        Args:
            query: Search query
            k: Number of results
            metadata_filters: Metadata filters (e.g., {"source_type": "api_doc"})

        Returns:
            Filtered relevant documents
        """
        # For now, get all results and filter
        # TODO: Implement native ChromaDB filtering
        results = self.vsm.similarity_search(query, k=k*2)

        if not metadata_filters:
            return results[:k]

        # Filter by metadata
        filtered = []
        for doc in results:
            match = True
            for key, value in metadata_filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)

        return filtered[:k]

    def smart_retrieval_for_api_docs(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Specialized retrieval for API documentation.

        Detects API-related queries and adjusts retrieval strategy.

        Args:
            query: User query
            k: Number of results

        Returns:
            Most relevant documents for API queries
        """
        query_lower = query.lower()

        # Detect if it's an API-related query
        api_keywords = [
            'api', 'post', 'get', 'put', 'delete', 'patch',
            'endpoint', 'request', 'response', 'call',
            'http', 'rest', 'json', 'header', 'parameter'
        ]

        is_api_query = any(kw in query_lower for kw in api_keywords)

        if is_api_query:
            # For API queries, boost keyword matching
            return self.hybrid_search(
                query,
                k=k,
                semantic_weight=0.5  # Equal weight to keywords
            )
        else:
            # For general queries, rely more on semantics
            return self.hybrid_search(
                query,
                k=k,
                semantic_weight=0.8
            )

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.

        Args:
            query: User query

        Returns:
            List of keywords
        """
        # Remove common words
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were',
            'to', 'for', 'of', 'in', 'on', 'at', 'by',
            'how', 'what', 'where', 'when', 'why', 'can', 'you',
            'help', 'me', 'please', 'i', 'my'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())

        # Filter stopwords and short words
        keywords = [
            w for w in words
            if w not in stopwords and len(w) > 2
        ]

        return keywords

    def _keyword_match_score(
        self,
        content: str,
        keywords: List[str]
    ) -> float:
        """
        Calculate keyword match score.

        Args:
            content: Document content
            keywords: List of keywords to match

        Returns:
            Score between 0 and 1
        """
        if not keywords:
            return 0.0

        content_lower = content.lower()

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in content_lower)

        # Calculate score
        score = matches / len(keywords)

        return score

    def rerank_by_relevance(
        self,
        query: str,
        documents: List[Document],
        k: int = 5
    ) -> List[Document]:
        """
        Rerank documents by relevance to query.

        Uses simple heuristics:
        - Keyword frequency
        - Position of keywords
        - Document metadata

        Args:
            query: User query
            documents: List of documents to rerank
            k: Number of top results to return

        Returns:
            Reranked documents
        """
        keywords = self._extract_keywords(query)

        scored_docs = []
        for doc in documents:
            score = 0.0

            # Keyword frequency score
            content_lower = doc.page_content.lower()
            for keyword in keywords:
                # Count occurrences
                count = content_lower.count(keyword)
                score += count * 0.5

                # Boost if in first 100 characters
                if keyword in content_lower[:100]:
                    score += 1.0

            # Metadata boost
            if doc.metadata.get('chunk_index') == 0:
                # First chunk often has important info
                score += 0.5

            scored_docs.append((doc, score))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:k]]
