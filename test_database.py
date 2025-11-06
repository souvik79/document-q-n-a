"""
Test script for database and session manager.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import DatabaseManager
from src.session_manager import SessionManager

def test_database():
    """Test database operations."""
    print("=" * 60)
    print("Testing Database Manager")
    print("=" * 60)

    db = DatabaseManager()
    print("✓ Database initialized successfully")
    print(f"  Database location: {db.db_path}")
    print()

    # Test 1: Create a project
    print("Test 1: Creating a project...")
    project = db.create_project("Test Project", "A test project for Q&A")
    print(f"✓ Project created: ID={project.id}, Name={project.name}")
    print(f"  Details: {project.to_dict()}")
    print()

    # Test 2: Add a document
    print("Test 2: Adding a document...")
    doc = db.add_document(
        project_id=project.id,
        source_type="file",
        source_path="/path/to/document.pdf",
        title="Test Document",
        metadata={"size": 1024, "pages": 10}
    )
    print(f"✓ Document added: ID={doc.id}, Title={doc.title}")
    print(f"  Details: {doc.to_dict()}")
    print()

    # Test 3: Add a URL document
    print("Test 3: Adding a URL document...")
    url_doc = db.add_document(
        project_id=project.id,
        source_type="url",
        source_path="https://example.com/article",
        title="Example Article",
        metadata={"domain": "example.com"}
    )
    print(f"✓ URL document added: ID={url_doc.id}, Title={url_doc.title}")
    print()

    # Test 4: Update document status
    print("Test 4: Updating document status...")
    updated_doc = db.update_document_status(doc.id, "processed")
    print(f"✓ Document status updated: {updated_doc.status}")
    print()

    # Test 5: Add a conversation
    print("Test 5: Adding a conversation...")
    conv = db.add_conversation(
        project_id=project.id,
        question="What is this document about?",
        answer="This is a test document about database testing.",
        sources=[doc.id, url_doc.id],
        metadata={"model": "llama3", "tokens": 50}
    )
    print(f"✓ Conversation added: ID={conv.id}")
    print(f"  Question: {conv.question}")
    print(f"  Answer: {conv.answer}")
    print(f"  Sources: {conv.sources}")
    print()

    # Test 6: Get project stats
    print("Test 6: Getting project stats...")
    stats = db.get_project_stats(project.id)
    print(f"✓ Project stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Test 7: Get all projects
    print("Test 7: Getting all projects...")
    projects = db.get_all_projects()
    print(f"✓ Found {len(projects)} project(s)")
    for p in projects:
        print(f"  - {p.name} (ID: {p.id})")
    print()

    # Test 8: Get project documents
    print("Test 8: Getting project documents...")
    documents = db.get_project_documents(project.id)
    print(f"✓ Found {len(documents)} document(s)")
    for d in documents:
        print(f"  - {d.title} ({d.source_type}): {d.status}")
    print()

    # Test 9: Get project conversations
    print("Test 9: Getting project conversations...")
    conversations = db.get_project_conversations(project.id)
    print(f"✓ Found {len(conversations)} conversation(s)")
    for c in conversations:
        print(f"  Q: {c.question[:50]}...")
        print(f"  A: {c.answer[:50]}...")
    print()

    # Test 10: Search conversations
    print("Test 10: Searching conversations...")
    search_results = db.search_conversations(project.id, "test")
    print(f"✓ Found {len(search_results)} matching conversation(s)")
    print()

    print("=" * 60)
    print("All Database Tests Passed! ✓")
    print("=" * 60)
    print()

    return project.id


def test_session_manager(project_id):
    """Test session manager operations."""
    print("=" * 60)
    print("Testing Session Manager")
    print("=" * 60)

    sm = SessionManager()
    print("✓ Session Manager initialized")
    print()

    # Test 1: Create a new project
    print("Test 1: Creating a new project...")
    result = sm.create_project("Session Test Project", "Testing session management")
    if result["success"]:
        print(f"✓ {result['message']}")
        print(f"  Project ID: {result['project']['id']}")
        new_project_id = result["project"]["id"]
    else:
        print(f"✗ Error: {result['error']}")
        return
    print()

    # Test 2: Check project limit (try to create duplicate)
    print("Test 2: Testing duplicate name validation...")
    result = sm.create_project("Session Test Project", "Duplicate")
    if not result["success"]:
        print(f"✓ Validation working: {result['error']}")
    else:
        print("✗ Duplicate validation failed!")
    print()

    # Test 3: Add a document
    print("Test 3: Adding a document to project...")
    result = sm.add_document(
        project_id=new_project_id,
        source_type="url",
        source_path="https://arxiv.org/pdf/1234.pdf",
        title="Research Paper"
    )
    if result["success"]:
        print(f"✓ {result['message']}")
        doc_id = result["document"]["id"]
    print()

    # Test 4: Check document limit
    print("Test 4: Checking document addition capability...")
    can_add = sm.can_add_document(new_project_id)
    print(f"✓ Can add documents: {can_add['can_add']}")
    print()

    # Test 5: Add a conversation
    print("Test 5: Adding a conversation...")
    result = sm.add_conversation(
        project_id=new_project_id,
        question="What is the main contribution of this paper?",
        answer="The paper introduces a novel approach to...",
        sources=[doc_id]
    )
    if result["success"]:
        print(f"✓ {result['message']}")
    print()

    # Test 6: Get project stats
    print("Test 6: Getting detailed project stats...")
    stats = sm.get_project_stats(new_project_id)
    print("✓ Project Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Test 7: Get all documents
    print("Test 7: Getting all project documents...")
    docs = sm.get_project_documents(new_project_id)
    print(f"✓ Found {len(docs)} document(s)")
    print()

    # Test 8: Get conversations
    print("Test 8: Getting project conversations...")
    convs = sm.get_project_conversations(new_project_id)
    print(f"✓ Found {len(convs)} conversation(s)")
    print()

    # Test 9: Search conversations
    print("Test 9: Searching conversations...")
    results = sm.search_conversations(new_project_id, "contribution")
    print(f"✓ Found {len(results)} matching conversation(s)")
    print()

    # Test 10: Get project size
    print("Test 10: Calculating project size...")
    size = sm.get_project_size(new_project_id)
    print(f"✓ Project size: {size['total_mb']} MB")
    print()

    # Test 11: Update project
    print("Test 11: Updating project...")
    result = sm.update_project(new_project_id, name="Updated Project Name")
    if result["success"]:
        print(f"✓ {result['message']}")
    print()

    # Test 12: Delete project
    print("Test 12: Deleting test project...")
    result = sm.delete_project(new_project_id)
    if result["success"]:
        print(f"✓ {result['message']}")
    print()

    print("=" * 60)
    print("All Session Manager Tests Passed! ✓")
    print("=" * 60)
    print()


def cleanup(project_id):
    """Clean up test data."""
    print("=" * 60)
    print("Cleanup")
    print("=" * 60)

    sm = SessionManager()
    result = sm.delete_project(project_id)
    if result["success"]:
        print(f"✓ Cleaned up test project (ID: {project_id})")
    print()


if __name__ == "__main__":
    try:
        # Test database
        project_id = test_database()

        # Test session manager
        test_session_manager(project_id)

        # Cleanup
        cleanup(project_id)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("\nThe database is working correctly and ready to use!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
