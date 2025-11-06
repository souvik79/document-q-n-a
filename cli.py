#!/usr/bin/env python3
"""
Command-line interface for Document Q&A System.
Provides interactive terminal access to all features.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.session_manager import SessionManager
from src.document_processor import DocumentProcessor
from src.web_scraper import WebScraper
from src.vector_store import VectorStoreManager
from src.qa_chain import QAChain
from src.config import validate_config


class DocumentQACLI:
    """Command-line interface for Document Q&A system."""

    # File to store last selected project
    PROJECT_FILE = Path(__file__).parent / "data" / ".last_project"

    def __init__(self):
        """Initialize CLI."""
        self.session_manager = SessionManager()
        self.current_project_id = self._load_last_project()
        self.qa_chain = None

    def _save_last_project(self, project_id: int):
        """Save the last selected project ID."""
        self.PROJECT_FILE.parent.mkdir(exist_ok=True)
        self.PROJECT_FILE.write_text(str(project_id))

    def _load_last_project(self) -> Optional[int]:
        """Load the last selected project ID."""
        if self.PROJECT_FILE.exists():
            try:
                return int(self.PROJECT_FILE.read_text().strip())
            except:
                return None
        return None

    def print_banner(self):
        """Print application banner."""
        print("\n" + "=" * 70)
        print(" " * 20 + "📚 Document Q&A System")
        print(" " * 15 + "AI-Powered Document Question Answering")
        print("=" * 70 + "\n")

    def validate_system(self):
        """Validate system configuration."""
        print("🔍 Validating system configuration...")
        config_status = validate_config()

        if config_status["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in config_status["warnings"]:
                print(f"  - {warning}")

        if not config_status["valid"]:
            print("\n❌ Configuration errors:")
            for error in config_status["errors"]:
                print(f"  - {error}")
            return False

        print("✅ System configuration valid\n")
        return True

    def list_projects(self):
        """List all projects."""
        projects = self.session_manager.get_all_projects()

        if not projects:
            print("📭 No projects found. Create one with: cli.py project create\n")
            return

        print(f"\n📂 Your Projects ({len(projects)}):")
        print("-" * 70)

        for project in projects:
            stats = self.session_manager.get_project_stats(project.id)
            last_accessed = datetime.fromisoformat(project.last_accessed.isoformat())
            time_ago = self._time_ago(last_accessed)

            # Mark active project
            marker = "→" if project.id == self.current_project_id else " "

            print(f"\n{marker} ID: {project.id}")
            print(f"  Name: {project.name}")
            if project.description:
                print(f"  Description: {project.description}")
            print(f"  Documents: {stats.get('total_documents', 0)} "
                  f"({stats.get('processed_documents', 0)} processed)")
            print(f"  Conversations: {stats.get('total_conversations', 0)}")
            print(f"  Last accessed: {time_ago}")

        print()
        if self.current_project_id:
            print(f"💡 Active project: {self.current_project_id} (marked with →)")
        else:
            print("💡 No active project. Select one with: python cli.py project select <id>")
        print()

    def create_project(self, name: str, description: Optional[str] = None):
        """Create a new project."""
        result = self.session_manager.create_project(name, description)

        if result["success"]:
            print(f"✅ {result['message']}")
            print(f"   Project ID: {result['project']['id']}")
        else:
            print(f"❌ {result['error']}")

    def select_project(self, project_id: int):
        """Select a project to work with."""
        project = self.session_manager.get_project(project_id)

        if not project:
            print(f"❌ Project with ID {project_id} not found")
            return False

        self.current_project_id = project_id
        self.qa_chain = QAChain(project_id)

        # Save for future commands
        self._save_last_project(project_id)

        print(f"✅ Selected project: {project.name} (ID: {project_id})")

        # Show project stats
        stats = self.session_manager.get_project_stats(project_id)
        print(f"   Documents: {stats.get('total_documents', 0)}")
        print(f"   Conversations: {stats.get('total_conversations', 0)}")
        print()

        return True

    def add_document(self, file_path: str):
        """Add a document file to current project."""
        if not self.current_project_id:
            print("❌ No project selected. Use: cli.py project select <id>")
            return

        file_path = Path(file_path).resolve()

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return

        print(f"📄 Processing file: {file_path.name}")

        try:
            # Process document
            processor = DocumentProcessor()

            print("   Validating file...")
            validation = processor.validate_file(file_path)
            if not validation["valid"]:
                print(f"❌ {validation['error']}")
                return

            print("   Extracting text and creating chunks...")
            chunks, metadata = processor.process_file(file_path)

            print(f"   ✓ Created {len(chunks)} chunks")

            # Add to database
            doc_result = self.session_manager.add_document(
                project_id=self.current_project_id,
                source_type="file",
                source_path=str(file_path),
                title=file_path.name,
                metadata=metadata
            )

            if not doc_result["success"]:
                print(f"❌ {doc_result['error']}")
                return

            document_id = doc_result["document"]["id"]

            # Add to vector store
            print("   Adding to vector database...")
            vs = VectorStoreManager(self.current_project_id)
            vs_result = vs.add_documents(chunks, document_id=document_id)

            if vs_result["success"]:
                # Update document status
                from src.database import get_db_manager
                db = get_db_manager()
                db.update_document_status(document_id, "processed")

                print(f"✅ Document added successfully!")
                print(f"   Document ID: {document_id}")
                print(f"   Chunks: {metadata['chunk_count']}")
                print(f"   Total characters: {metadata['total_characters']}")
            else:
                print(f"❌ Failed to add to vector store: {vs_result['error']}")

        except Exception as e:
            print(f"❌ Error processing document: {e}")

    def add_url(self, url: str):
        """Add a URL to current project."""
        if not self.current_project_id:
            print("❌ No project selected. Use: cli.py project select <id>")
            return

        print(f"🌐 Processing URL: {url}")

        try:
            # Process URL
            scraper = WebScraper()

            print("   Validating URL...")
            validation = scraper.validate_url(url)
            if not validation["valid"]:
                print(f"❌ {validation['error']}")
                return

            print("   Fetching content...")
            chunks, metadata = scraper.process_url(url, self.current_project_id)

            print(f"   ✓ Created {len(chunks)} chunks")

            # Add to database
            doc_result = self.session_manager.add_document(
                project_id=self.current_project_id,
                source_type="url",
                source_path=url,
                title=metadata.get("title", url),
                metadata=metadata
            )

            if not doc_result["success"]:
                print(f"❌ {doc_result['error']}")
                return

            document_id = doc_result["document"]["id"]

            # Add to vector store
            print("   Adding to vector database...")
            vs = VectorStoreManager(self.current_project_id)
            vs_result = vs.add_documents(chunks, document_id=document_id)

            if vs_result["success"]:
                # Update document status
                from src.database import get_db_manager
                db = get_db_manager()
                db.update_document_status(document_id, "processed")

                print(f"✅ URL added successfully!")
                print(f"   Document ID: {document_id}")
                print(f"   Title: {metadata.get('title', 'N/A')}")
                print(f"   Chunks: {metadata['chunk_count']}")
            else:
                print(f"❌ Failed to add to vector store: {vs_result['error']}")

        except Exception as e:
            print(f"❌ Error processing URL: {e}")

    def list_documents(self):
        """List documents in current project."""
        if not self.current_project_id:
            print("❌ No project selected. Use: cli.py project select <id>")
            return

        documents = self.session_manager.get_project_documents(self.current_project_id)

        if not documents:
            print("📭 No documents in this project\n")
            return

        print(f"\n📚 Documents in Project ({len(documents)}):")
        print("-" * 70)

        for doc in documents:
            print(f"\n  ID: {doc['id']}")
            print(f"  Title: {doc['title']}")
            print(f"  Type: {doc['source_type']}")
            print(f"  Status: {doc['status']}")
            if doc['source_type'] == 'file':
                print(f"  Path: {doc['source_path']}")
            else:
                print(f"  URL: {doc['source_path'][:60]}...")

            created_at = datetime.fromisoformat(doc['created_at'])
            print(f"  Added: {self._time_ago(created_at)}")

        print()

    def ask_question(self, question: str):
        """Ask a question about the documents."""
        if not self.current_project_id:
            print("❌ No project selected. Use: cli.py project select <id>")
            return

        if not self.qa_chain:
            self.qa_chain = QAChain(self.current_project_id)

        print(f"\n💬 Question: {question}")
        print("-" * 70)
        print("🤔 Thinking...\n")

        try:
            result = self.qa_chain.ask(question)

            if result["success"]:
                print("🤖 Answer:")
                print(result["answer"])
                print()

                print("📊 Metadata:")
                print(f"   Model: {result['metadata']['model']}")
                print(f"   Query time: {result['metadata']['query_time_seconds']}s")
                print(f"   Sources used: {result['metadata']['num_sources']}")
                print()

                if result['sources']:
                    print("📄 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"   {i}. {source['content'][:100]}...")

                # Save conversation
                self.session_manager.add_conversation(
                    project_id=self.current_project_id,
                    question=question,
                    answer=result["answer"],
                    sources=result.get("source_document_ids", []),
                    metadata=result["metadata"]
                )

                print("\n✅ Conversation saved")
            else:
                print(f"❌ {result['error']}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def interactive_chat(self):
        """Start interactive chat mode."""
        if not self.current_project_id:
            print("❌ No project selected. Use: cli.py project select <id>")
            return

        if not self.qa_chain:
            self.qa_chain = QAChain(self.current_project_id)

        print("\n💬 Interactive Chat Mode")
        print("-" * 70)
        print("Type your questions (or 'quit' to exit, 'history' to see past conversations)")
        print()

        while True:
            try:
                question = input("You: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if question.lower() == 'history':
                    self.show_conversations()
                    continue

                print("🤖 Assistant: ", end="", flush=True)

                result = self.qa_chain.ask(question)

                if result["success"]:
                    print(result["answer"])

                    # Save conversation
                    self.session_manager.add_conversation(
                        project_id=self.current_project_id,
                        question=question,
                        answer=result["answer"],
                        sources=result.get("source_document_ids", []),
                        metadata=result["metadata"]
                    )
                else:
                    print(f"Error: {result['error']}")

                print()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def show_conversations(self):
        """Show conversation history for current project."""
        if not self.current_project_id:
            print("❌ No project selected")
            return

        conversations = self.session_manager.get_project_conversations(
            self.current_project_id,
            limit=10
        )

        if not conversations:
            print("📭 No conversations yet\n")
            return

        print(f"\n💬 Recent Conversations (last {len(conversations)}):")
        print("-" * 70)

        for conv in conversations[-5:]:  # Show last 5
            created_at = datetime.fromisoformat(conv['created_at'])
            print(f"\n  [{self._time_ago(created_at)}]")
            print(f"  Q: {conv['question']}")
            print(f"  A: {conv['answer'][:100]}...")

        print()

    def show_stats(self):
        """Show statistics for current project."""
        if not self.current_project_id:
            print("❌ No project selected")
            return

        stats = self.session_manager.get_project_stats(self.current_project_id)
        size = self.session_manager.get_project_size(self.current_project_id)

        print(f"\n📊 Project Statistics:")
        print("-" * 70)
        print(f"  Total documents: {stats.get('total_documents', 0)}")
        print(f"  Processed: {stats.get('processed_documents', 0)}")
        print(f"  Failed: {stats.get('failed_documents', 0)}")
        print(f"  Total conversations: {stats.get('total_conversations', 0)}")
        print(f"  Storage size: {size['total_mb']} MB")

        # Vector store stats
        if self.qa_chain:
            qa_stats = self.qa_chain.get_stats()
            print(f"  Vector chunks: {qa_stats.get('total_chunks', 0)}")
            print(f"  Model: {qa_stats.get('model', 'N/A')}")

        print()

    def delete_project(self, project_id: int):
        """Delete a project."""
        project = self.session_manager.get_project(project_id)
        if not project:
            print(f"❌ Project {project_id} not found")
            return

        confirm = input(f"⚠️  Delete project '{project.name}'? This cannot be undone! (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Cancelled")
            return

        result = self.session_manager.delete_project(project_id)
        if result["success"]:
            print(f"✅ {result['message']}")
            if self.current_project_id == project_id:
                self.current_project_id = None
                self.qa_chain = None
        else:
            print(f"❌ {result['error']}")

    @staticmethod
    def _time_ago(dt: datetime) -> str:
        """Convert datetime to human-readable time ago string."""
        now = datetime.utcnow()
        diff = now - dt

        seconds = diff.total_seconds()

        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days}d ago"
        else:
            return dt.strftime("%Y-%m-%d")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Document Q&A System - AI-powered document question answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all projects
  %(prog)s project list

  # Create a new project
  %(prog)s project create "Research Papers" --description "ML research papers"

  # Select a project
  %(prog)s project select 1

  # Add a document
  %(prog)s document add /path/to/document.pdf

  # Add a URL
  %(prog)s document add-url https://example.com/article

  # Ask a question
  %(prog)s ask "What is machine learning?"

  # Start interactive chat
  %(prog)s chat

  # Show project statistics
  %(prog)s stats
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Project commands
    project_parser = subparsers.add_parser('project', help='Project management')
    project_sub = project_parser.add_subparsers(dest='subcommand')

    project_sub.add_parser('list', help='List all projects')

    create_parser = project_sub.add_parser('create', help='Create new project')
    create_parser.add_argument('name', help='Project name')
    create_parser.add_argument('--description', '-d', help='Project description')

    select_parser = project_sub.add_parser('select', help='Select a project')
    select_parser.add_argument('id', type=int, help='Project ID')

    delete_parser = project_sub.add_parser('delete', help='Delete a project')
    delete_parser.add_argument('id', type=int, help='Project ID')

    # Document commands
    doc_parser = subparsers.add_parser('document', help='Document management')
    doc_sub = doc_parser.add_subparsers(dest='subcommand')

    doc_sub.add_parser('list', help='List documents in current project')

    add_parser = doc_sub.add_parser('add', help='Add a document file')
    add_parser.add_argument('file', help='Path to document file')

    url_parser = doc_sub.add_parser('add-url', help='Add a URL')
    url_parser.add_argument('url', help='URL to add')

    # Q&A commands
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', help='Your question')

    subparsers.add_parser('chat', help='Start interactive chat mode')

    subparsers.add_parser('history', help='Show conversation history')

    subparsers.add_parser('stats', help='Show project statistics')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = DocumentQACLI()
    cli.print_banner()

    if not cli.validate_system():
        sys.exit(1)

    # Execute command
    if args.command == 'project':
        if args.subcommand == 'list':
            cli.list_projects()
        elif args.subcommand == 'create':
            cli.create_project(args.name, args.description)
        elif args.subcommand == 'select':
            cli.select_project(args.id)
        elif args.subcommand == 'delete':
            cli.delete_project(args.id)

    elif args.command == 'document':
        if args.subcommand == 'list':
            cli.list_documents()
        elif args.subcommand == 'add':
            cli.add_document(args.file)
        elif args.subcommand == 'add-url':
            cli.add_url(args.url)

    elif args.command == 'ask':
        cli.ask_question(args.question)

    elif args.command == 'chat':
        cli.interactive_chat()

    elif args.command == 'history':
        cli.show_conversations()

    elif args.command == 'stats':
        cli.show_stats()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
