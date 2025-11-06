"""
Test script for web scraper.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.web_scraper import WebScraper, is_valid_url

def main():
    print("=" * 60)
    print("Web Scraper Test")
    print("=" * 60)
    print()

    scraper = WebScraper()

    # Test 1: URL Validation
    print("Test 1: URL Validation")
    print("-" * 60)

    test_urls = [
        ("https://www.python.org", "Valid HTTPS URL"),
        ("http://example.com", "Valid HTTP URL"),
        ("https://en.wikipedia.org/wiki/Python", "Wikipedia article"),
        ("not-a-url", "Invalid URL"),
        ("ftp://files.example.com", "Unsupported protocol"),
        ("", "Empty URL"),
    ]

    for url, description in test_urls:
        validation = scraper.validate_url(url)
        status = "✓" if validation['valid'] else "✗"
        print(f"{status} {description}: {url}")
        if not validation['valid']:
            print(f"   Error: {validation['error']}")
        else:
            print(f"   Domain: {validation['domain']}")
    print()

    # Test 2: Quick validation function
    print("Test 2: Quick URL Validation Function")
    print("-" * 60)
    quick_tests = [
        "https://github.com",
        "invalid url",
        "https://docs.python.org/3/",
    ]
    for url in quick_tests:
        result = is_valid_url(url)
        print(f"  {url}: {'✓ Valid' if result else '✗ Invalid'}")
    print()

    # Test 3: Fetch URL metadata
    print("Test 3: Fetch URL Metadata")
    print("-" * 60)
    test_url = "https://www.python.org"
    print(f"Fetching metadata from: {test_url}")

    metadata = scraper.get_url_metadata(test_url)
    if 'error' in metadata:
        print(f"✗ Error: {metadata['error']}")
    else:
        print("✓ Metadata fetched successfully:")
        print(f"  Title: {metadata.get('title', 'N/A')}")
        print(f"  Domain: {metadata.get('domain', 'N/A')}")
        print(f"  Description: {metadata.get('description', 'N/A')[:100]}...")
        print(f"  Content length: {metadata.get('content_length', 0):,} characters")
        print(f"  Content type: {metadata.get('content_type', 'N/A')}")
    print()

    # Test 4: Process URL (full pipeline)
    print("Test 4: Complete URL Processing")
    print("-" * 60)
    process_url = "https://www.python.org"
    print(f"Processing: {process_url}")
    print()

    try:
        chunks, process_metadata = scraper.process_url(process_url)
        print("✓ Processing successful!")
        print()
        print("Processing Results:")
        for key, value in process_metadata.items():
            if key != 'cache_path':
                print(f"  {key}: {value}")
        print()

        print(f"Chunks created: {len(chunks)}")
        print()

        # Show first chunk
        if chunks:
            print("First chunk preview:")
            print("-" * 40)
            preview = chunks[0].page_content[:300]
            print(preview)
            if len(chunks[0].page_content) > 300:
                print("...")
            print("-" * 40)
            print()

            print("Chunk metadata:")
            for key, value in chunks[0].metadata.items():
                if key != 'description':
                    print(f"  {key}: {value}")
                else:
                    desc = str(value)[:100] if value else "None"
                    print(f"  {key}: {desc}...")

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test 5: Text preview extraction
    print("Test 5: Extract Text Preview")
    print("-" * 60)
    preview = scraper.extract_text_preview("https://www.python.org", max_chars=200)
    print(preview)
    print()

    print("=" * 60)
    print("Web Scraper Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
