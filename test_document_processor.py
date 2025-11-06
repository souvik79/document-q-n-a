"""
Test script for document processor.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor, get_supported_extensions, is_supported_file

def main():
    print("=" * 60)
    print("Document Processor Test")
    print("=" * 60)
    print()

    # Show supported extensions
    print("Supported file extensions:")
    for ext in get_supported_extensions():
        print(f"  - .{ext}")
    print()

    processor = DocumentProcessor()

    # Create a test directory and file
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)

    # Create a test text file
    test_file = test_dir / "test_document.txt"
    with open(test_file, 'w') as f:
        f.write("""# Sample Document for Testing

This is a comprehensive test document for the Document Q&A system.

## Introduction
This system allows users to upload documents and ask questions about them.
The system uses LangChain for document processing and Ollama (Llama3) for answering questions.

## Key Features

### Document Processing
- Support for multiple document formats: PDF, DOCX, TXT, MD, CSV
- Automatic text extraction and chunking
- Metadata preservation
- File validation and size limits

### Web URL Processing
- Fetch content from web pages
- Extract and process HTML content
- Cache web content for offline access

### Q&A System
- Natural language question answering
- Source attribution - see which documents were used
- Conversation history tracking
- Multi-turn conversations

### Data Management
- Multiple projects (knowledge bases)
- Persistent storage with SQLite
- Vector database with ChromaDB
- Session management

### Export & Search
- Export conversations to JSON, CSV, PDF
- Search within conversation history
- Full-text search across documents

## Technical Stack

- **LLM**: Ollama with Llama3
- **Framework**: LangChain for document processing
- **Vector DB**: ChromaDB for semantic search
- **Database**: SQLite for metadata and conversations
- **UI**: Streamlit for web interface
- **Monitoring**: LangSmith for AI call tracking

## How It Works

1. **Upload Documents**: Users upload PDF, DOCX, TXT files or provide URLs
2. **Processing**: Documents are split into chunks and embedded
3. **Storage**: Chunks are stored in ChromaDB vector database
4. **Query**: User asks a question
5. **Retrieval**: Relevant chunks are retrieved using similarity search
6. **Generation**: Llama3 generates an answer based on retrieved context
7. **Response**: Answer is displayed with source citations

## Use Cases

- Research paper analysis
- Documentation Q&A systems
- Knowledge base queries
- Contract analysis
- Meeting notes summarization
- Educational material comprehension

## Conclusion

This is a powerful, local-first document Q&A system that preserves privacy
while providing accurate answers based on your documents.

The system is designed to be extensible and can be enhanced with additional
features like multi-user support, advanced analytics, and more sophisticated
retrieval strategies in future versions.
""")

    print(f"Created test file: {test_file}")
    print(f"File size: {test_file.stat().st_size} bytes")
    print()

    # Test 1: File validation
    print("Test 1: File Validation")
    print("-" * 60)
    validation = processor.validate_file(test_file)
    print(f"Valid: {validation['valid']}")
    if validation['valid']:
        print(f"File size: {validation['file_size']} bytes")
        print(f"Extension: .{validation['file_extension']}")
        print(f"MIME type: {validation['mime_type']}")
    else:
        print(f"Error: {validation['error']}")
    print()

    # Test 2: Metadata extraction
    print("Test 2: Metadata Extraction")
    print("-" * 60)
    metadata = processor.get_file_metadata(test_file)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print()

    # Test 3: File type checking
    print("Test 3: File Type Checking")
    print("-" * 60)
    test_filenames = [
        "document.pdf",
        "report.docx",
        "notes.txt",
        "readme.md",
        "data.csv",
        "image.jpg",  # Not supported
    ]
    for filename in test_filenames:
        supported = is_supported_file(filename)
        status = "✓ Supported" if supported else "✗ Not supported"
        print(f"{filename}: {status}")
    print()

    # Test 4: Document processing
    print("Test 4: Complete Document Processing")
    print("-" * 60)
    try:
        chunks, process_metadata = processor.process_file(test_file)
        print("✓ Processing successful!")
        print(f"\nProcessing Results:")
        for key, value in process_metadata.items():
            print(f"  {key}: {value}")

        print(f"\nChunks created: {len(chunks)}")

        # Show first few chunks
        if chunks:
            print(f"\nFirst chunk preview:")
            print("-" * 40)
            print(chunks[0].page_content[:300])
            if len(chunks[0].page_content) > 300:
                print("...")
            print("-" * 40)

            print(f"\nChunk metadata:")
            for key, value in chunks[0].metadata.items():
                print(f"  {key}: {value}")

        # Get statistics
        print(f"\nDocument Statistics:")
        stats = processor.get_document_stats(chunks)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 5: Text preview
    print("Test 5: Text Preview Extraction")
    print("-" * 60)
    preview = processor.extract_text_preview(test_file, max_chars=200)
    print(preview)
    print()

    # Cleanup
    print("=" * 60)
    print("Cleaning up test files...")
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print("✓ Cleanup complete")

    print("\n" + "=" * 60)
    print("Document Processor Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
