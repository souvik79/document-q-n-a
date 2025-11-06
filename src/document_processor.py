"""
Document processor for handling file uploads and text extraction.
Supports PDF, DOCX, TXT, MD, CSV and other text-based formats.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_FILE_TYPES,
    get_uploads_dir,
)
from .image_processor import ImageProcessor


class DocumentProcessor:
    """Handles document loading, validation, and text extraction."""

    def __init__(self, use_vision_for_images: bool = True):
        """Initialize document processor.

        Args:
            use_vision_for_images: Whether to use vision models for image understanding
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Initialize image processor
        self.image_processor = ImageProcessor(
            use_tesseract=True,
            use_easyocr=False,  # Tesseract is faster
            use_vision_model=use_vision_for_images,
            vision_model="llava:7b"
        )

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate file before processing.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with validation result
        """
        # Check if file exists
        if not file_path.exists():
            return {"valid": False, "error": "File does not exist"}

        # Check if it's a file (not directory)
        if not file_path.is_file():
            return {"valid": False, "error": "Path is not a file"}

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            return {
                "valid": False,
                "error": f"File size ({actual_mb:.2f} MB) exceeds maximum allowed size ({max_mb} MB)"
            }

        # Check file type
        file_extension = file_path.suffix.lower().lstrip('.')
        if file_extension not in SUPPORTED_FILE_TYPES:
            supported = ", ".join(SUPPORTED_FILE_TYPES.keys())
            return {
                "valid": False,
                "error": f"Unsupported file type '.{file_extension}'. Supported types: {supported}"
            }

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))

        return {
            "valid": True,
            "file_size": file_size,
            "file_extension": file_extension,
            "mime_type": mime_type or "application/octet-stream",
        }

    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file metadata
        """
        stat = file_path.stat()

        return {
            "filename": file_path.name,
            "file_size": stat.st_size,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix.lower(),
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
        }

    def save_uploaded_file(self, uploaded_file, project_id: int) -> Tuple[Path, Dict[str, Any]]:
        """
        Save an uploaded file to the project's uploads directory.

        Args:
            uploaded_file: Streamlit UploadedFile object or file-like object
            project_id: Project ID

        Returns:
            Tuple of (saved file path, metadata)
        """
        uploads_dir = get_uploads_dir(project_id)

        # Generate unique filename using hash to avoid conflicts
        original_name = uploaded_file.name if hasattr(uploaded_file, 'name') else "unknown_file"
        file_extension = Path(original_name).suffix

        # Create a hash of the file content for uniqueness
        file_content = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file
        file_hash = hashlib.md5(file_content).hexdigest()[:8]

        # Reset file pointer if it's a file-like object
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)

        # Create unique filename
        safe_name = "".join(c for c in Path(original_name).stem if c.isalnum() or c in (' ', '-', '_'))
        unique_filename = f"{safe_name}_{file_hash}{file_extension}"
        file_path = uploads_dir / unique_filename

        # Save file
        with open(file_path, 'wb') as f:
            if hasattr(uploaded_file, 'read'):
                f.write(uploaded_file.read())
            else:
                f.write(file_content)

        # Reset file pointer again
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)

        metadata = self.get_file_metadata(file_path)
        metadata["original_name"] = original_name

        return file_path, metadata

    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load and extract text from a document.

        Args:
            file_path: Path to the document

        Returns:
            List of LangChain Document objects

        Raises:
            ValueError: If file type is not supported or loading fails
        """
        file_extension = file_path.suffix.lower().lstrip('.')

        try:
            # Check if it's an image file
            if self.image_processor.is_image_file(str(file_path)):
                # Process image to extract text and description
                image_text = self.image_processor.process_and_create_document_text(
                    str(file_path),
                    title=file_path.stem
                )

                # Create a Document object
                doc = Document(
                    page_content=image_text,
                    metadata={
                        "source_file": file_path.name,
                        "file_path": str(file_path),
                        "file_type": file_extension,
                        "content_type": "image",
                    }
                )
                return [doc]

            # Select appropriate loader based on file type
            if file_extension == 'pdf':
                loader = PyPDFLoader(str(file_path))

            elif file_extension in ['docx', 'doc']:
                loader = Docx2txtLoader(str(file_path))

            elif file_extension == 'txt':
                loader = TextLoader(str(file_path), encoding='utf-8')

            elif file_extension == 'md':
                try:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                except:
                    # Fallback to text loader if unstructured is not available
                    loader = TextLoader(str(file_path), encoding='utf-8')

            elif file_extension == 'csv':
                loader = CSVLoader(str(file_path))

            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Load documents
            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source_file": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_extension,
                })

            return documents

        except Exception as e:
            raise ValueError(f"Failed to load document: {str(e)}")

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

    def process_file(self, file_path: Path) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Complete processing pipeline for a file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (list of chunked documents, processing metadata)
        """
        # Validate file
        validation = self.validate_file(file_path)
        if not validation["valid"]:
            raise ValueError(validation["error"])

        # Load document
        documents = self.load_document(file_path)

        # Count pages/sections
        page_count = len(documents)

        # Split into chunks
        chunks = self.split_documents(documents)

        # Calculate statistics
        total_chars = sum(len(chunk.page_content) for chunk in chunks)

        metadata = {
            "status": "processed",
            "page_count": page_count,
            "chunk_count": len(chunks),
            "total_characters": total_chars,
            "file_size": validation["file_size"],
            "file_extension": validation["file_extension"],
            "mime_type": validation["mime_type"],
        }

        return chunks, metadata

    def extract_text_preview(self, file_path: Path, max_chars: int = 500) -> str:
        """
        Extract a preview of the document text.

        Args:
            file_path: Path to the file
            max_chars: Maximum characters to extract

        Returns:
            Preview text
        """
        try:
            documents = self.load_document(file_path)
            if documents:
                text = documents[0].page_content[:max_chars]
                if len(documents[0].page_content) > max_chars:
                    text += "..."
                return text
            return "No text extracted"
        except:
            return "Failed to extract preview"

    def get_document_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.

        Args:
            chunks: List of document chunks

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "chunk_count": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        return {
            "chunk_count": len(chunks),
            "total_characters": sum(chunk_sizes),
            "avg_chunk_size": round(sum(chunk_sizes) / len(chunk_sizes)),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
        }


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return list(SUPPORTED_FILE_TYPES.keys())


def is_supported_file(filename: str) -> bool:
    """
    Check if a file is supported based on its extension.

    Args:
        filename: Name or path of the file

    Returns:
        True if supported, False otherwise
    """
    extension = Path(filename).suffix.lower().lstrip('.')
    return extension in SUPPORTED_FILE_TYPES


if __name__ == "__main__":
    # Test document processor
    print("=" * 60)
    print("Document Processor Test")
    print("=" * 60)

    processor = DocumentProcessor()

    # Test with a sample text file
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)

    # Create a test file
    test_file = test_dir / "test_document.txt"
    with open(test_file, 'w') as f:
        f.write("""# Sample Document

This is a test document for the Document Q&A system.

## Introduction
This system allows users to upload documents and ask questions about them.
The system uses LangChain for document processing and Ollama for answering questions.

## Features
- Support for multiple document formats (PDF, DOCX, TXT, MD)
- Web URL processing
- Conversation history
- Source attribution
- Export functionality

## Conclusion
This is a comprehensive document Q&A system built with modern AI technologies.
""")

    print(f"Created test file: {test_file}")
    print()

    # Test validation
    print("Testing file validation...")
    validation = processor.validate_file(test_file)
    print(f"  Valid: {validation['valid']}")
    if validation['valid']:
        print(f"  File size: {validation['file_size']} bytes")
        print(f"  Extension: {validation['file_extension']}")
    print()

    # Test metadata extraction
    print("Testing metadata extraction...")
    metadata = processor.get_file_metadata(test_file)
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()

    # Test document processing
    print("Testing document processing...")
    try:
        chunks, process_metadata = processor.process_file(test_file)
        print(f"  ✓ Processing successful!")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Total characters: {process_metadata['total_characters']}")
        print()

        # Show first chunk
        if chunks:
            print("First chunk preview:")
            print("-" * 60)
            print(chunks[0].page_content[:200])
            print("-" * 60)
            print()

        # Show statistics
        stats = processor.get_document_stats(chunks)
        print("Document statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"  ✗ Processing failed: {e}")

    print()
    print("=" * 60)
    print("Document Processor Test Complete!")
    print("=" * 60)

    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
