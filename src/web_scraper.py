"""
Web scraper for fetching and processing content from URLs.
Supports web pages, articles, and online documents.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from datetime import datetime
import time

import requests
from bs4 import BeautifulSoup
import validators

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_web_cache_dir,
)


class WebScraper:
    """Handles web URL fetching, content extraction, and processing."""

    def __init__(self):
        """Initialize web scraper."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate a URL before fetching.

        Args:
            url: URL to validate

        Returns:
            Dictionary with validation result
        """
        # Basic URL validation
        if not url or not url.strip():
            return {"valid": False, "error": "URL cannot be empty"}

        url = url.strip()

        # Validate URL format
        if not validators.url(url):
            return {"valid": False, "error": "Invalid URL format"}

        # Parse URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {"valid": False, "error": "URL must have scheme (http/https) and domain"}

            # Only allow http and https
            if parsed.scheme not in ['http', 'https']:
                return {"valid": False, "error": "Only HTTP and HTTPS protocols are supported"}

            return {
                "valid": True,
                "url": url,
                "scheme": parsed.scheme,
                "domain": parsed.netloc,
                "path": parsed.path,
            }

        except Exception as e:
            return {"valid": False, "error": f"URL parsing failed: {str(e)}"}

    def fetch_url(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Dictionary with fetched content and metadata
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()

            # Get final URL after redirects
            final_url = response.url

            # Get content type
            content_type = response.headers.get('content-type', 'unknown')

            # Check if content is HTML/text
            if 'text/html' not in content_type.lower() and 'text/plain' not in content_type.lower():
                return {
                    "success": False,
                    "error": f"Unsupported content type: {content_type}. Only HTML and text content is supported."
                }

            return {
                "success": True,
                "content": response.text,
                "final_url": final_url,
                "status_code": response.status_code,
                "content_type": content_type,
                "content_length": len(response.text),
                "headers": dict(response.headers),
            }

        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timeout after {timeout} seconds"}

        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Failed to connect to the URL"}

        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP error: {e.response.status_code}"}

        except Exception as e:
            return {"success": False, "error": f"Failed to fetch URL: {str(e)}"}

    def extract_text_from_html(self, html_content: str, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract clean text from HTML content.

        Args:
            html_content: HTML content string
            url: Source URL (for metadata)

        Returns:
            Tuple of (extracted text, metadata)
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                script.decompose()

            # Try to extract title
            title = None
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else None

            # Try to get page description
            description = None
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc.get('content').strip()

            # Extract main content
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body

            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Clean up text - remove excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)

            metadata = {
                "title": title,
                "description": description,
                "url": url,
                "text_length": len(text),
                "line_count": len(lines),
            }

            return text, metadata

        except Exception as e:
            raise ValueError(f"Failed to extract text from HTML: {str(e)}")

    def save_cached_content(self, url: str, content: str, project_id: int) -> Path:
        """
        Save fetched content to cache.

        Args:
            url: Source URL
            content: Content to cache
            project_id: Project ID

        Returns:
            Path to cached file
        """
        cache_dir = get_web_cache_dir(project_id)

        # Generate filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = cache_dir / f"{url_hash}.txt"

        # Save content with metadata
        cache_data = {
            "url": url,
            "fetched_at": datetime.utcnow().isoformat(),
            "content": content,
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        return cache_file

    def load_from_url(self, url: str) -> List[Document]:
        """
        Load and extract content from a URL using LangChain's WebBaseLoader.

        Args:
            url: URL to load

        Returns:
            List of LangChain Document objects

        Raises:
            ValueError: If loading fails
        """
        try:
            # Use LangChain's WebBaseLoader
            loader = WebBaseLoader(url)
            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_url": url,
                    "source_type": "url",
                    "fetched_at": datetime.utcnow().isoformat(),
                })

            return documents

        except Exception as e:
            # Fallback to manual scraping if WebBaseLoader fails
            try:
                fetch_result = self.fetch_url(url)
                if not fetch_result["success"]:
                    raise ValueError(fetch_result["error"])

                text, metadata = self.extract_text_from_html(fetch_result["content"], url)

                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        "source_url": url,
                        "source_type": "url",
                        "title": metadata.get("title"),
                        "description": metadata.get("description"),
                        "fetched_at": datetime.utcnow().isoformat(),
                    }
                )

                return [doc]

            except Exception as fallback_error:
                raise ValueError(f"Failed to load URL: {str(e)}. Fallback also failed: {str(fallback_error)}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)

            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_count"] = len(chunks)

            return chunks

        except Exception as e:
            raise ValueError(f"Failed to split documents: {str(e)}")

    def process_url(self, url: str, project_id: Optional[int] = None) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Complete processing pipeline for a URL.

        Args:
            url: URL to process
            project_id: Project ID (optional, for caching)

        Returns:
            Tuple of (list of chunked documents, processing metadata)
        """
        # Validate URL
        validation = self.validate_url(url)
        if not validation["valid"]:
            raise ValueError(validation["error"])

        # Load document
        start_time = time.time()
        documents = self.load_from_url(url)
        fetch_time = time.time() - start_time

        # Extract title from first document
        title = documents[0].metadata.get('title') or documents[0].metadata.get('source_url', url)

        # Cache content if project_id provided
        if project_id and documents:
            content = documents[0].page_content
            cache_path = self.save_cached_content(url, content, project_id)
        else:
            cache_path = None

        # Split into chunks
        chunks = self.split_documents(documents)

        # Calculate statistics
        total_chars = sum(len(chunk.page_content) for chunk in chunks)

        metadata = {
            "status": "processed",
            "url": url,
            "domain": validation["domain"],
            "title": title,
            "chunk_count": len(chunks),
            "total_characters": total_chars,
            "fetch_time_seconds": round(fetch_time, 2),
            "cached": cache_path is not None,
            "cache_path": str(cache_path) if cache_path else None,
        }

        return chunks, metadata

    def extract_text_preview(self, url: str, max_chars: int = 500) -> str:
        """
        Extract a preview of the URL content.

        Args:
            url: URL to preview
            max_chars: Maximum characters to extract

        Returns:
            Preview text
        """
        try:
            documents = self.load_from_url(url)
            if documents:
                text = documents[0].page_content[:max_chars]
                if len(documents[0].page_content) > max_chars:
                    text += "..."
                return text
            return "No text extracted"
        except Exception as e:
            return f"Failed to extract preview: {str(e)}"

    def get_url_metadata(self, url: str) -> Dict[str, Any]:
        """
        Get metadata from a URL without full processing.

        Args:
            url: URL to analyze

        Returns:
            Dictionary with URL metadata
        """
        validation = self.validate_url(url)
        if not validation["valid"]:
            return {"error": validation["error"]}

        try:
            fetch_result = self.fetch_url(url, timeout=10)
            if not fetch_result["success"]:
                return {"error": fetch_result["error"]}

            soup = BeautifulSoup(fetch_result["content"], 'html.parser')

            # Extract metadata
            title = soup.title.string.strip() if soup.title and soup.title.string else None

            description = None
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc.get('content').strip()

            return {
                "url": url,
                "domain": validation["domain"],
                "title": title,
                "description": description,
                "content_length": fetch_result["content_length"],
                "content_type": fetch_result["content_type"],
            }

        except Exception as e:
            return {"error": f"Failed to get metadata: {str(e)}"}


def is_valid_url(url: str) -> bool:
    """
    Quick check if a URL is valid.

    Args:
        url: URL to check

    Returns:
        True if valid, False otherwise
    """
    return validators.url(url) is True


if __name__ == "__main__":
    # Test web scraper
    print("=" * 60)
    print("Web Scraper Test")
    print("=" * 60)
    print()

    scraper = WebScraper()

    # Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://python.org",
        "invalid-url",
        "ftp://example.com",  # Unsupported protocol
    ]

    for url in test_urls:
        print(f"Testing URL: {url}")
        print("-" * 60)

        # Validate
        validation = scraper.validate_url(url)
        print(f"Valid: {validation['valid']}")

        if validation['valid']:
            print(f"Domain: {validation['domain']}")

            # Get metadata
            print("\nFetching metadata...")
            metadata = scraper.get_url_metadata(url)
            if 'error' not in metadata:
                print(f"  Title: {metadata.get('title', 'N/A')}")
                print(f"  Description: {metadata.get('description', 'N/A')[:100]}...")
                print(f"  Content length: {metadata.get('content_length', 0)} characters")

                # Process URL (only for first URL to save time)
                if url == test_urls[0]:
                    print("\nProcessing URL...")
                    try:
                        chunks, process_metadata = scraper.process_url(url)
                        print(f"  ✓ Processing successful!")
                        print(f"  Chunks created: {len(chunks)}")
                        print(f"  Total characters: {process_metadata['total_characters']}")
                        print(f"  Fetch time: {process_metadata['fetch_time_seconds']}s")

                        # Show first chunk preview
                        if chunks:
                            print(f"\n  First chunk preview:")
                            print("  " + "-" * 56)
                            preview = chunks[0].page_content[:200].replace('\n', '\n  ')
                            print(f"  {preview}...")
                            print("  " + "-" * 56)

                    except Exception as e:
                        print(f"  ✗ Processing failed: {e}")
            else:
                print(f"  Error: {metadata['error']}")
        else:
            print(f"Error: {validation['error']}")

        print()

    print("=" * 60)
    print("Web Scraper Test Complete!")
    print("=" * 60)
