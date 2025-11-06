"""
Session and project management module.
Handles project lifecycle, validation, and cleanup operations.
"""

import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path

from .database import DatabaseManager, Project, get_db_manager
from .config import (
    MAX_PROJECTS_PER_USER,
    MAX_DOCUMENTS_PER_PROJECT,
    get_project_dir,
    get_uploads_dir,
    get_web_cache_dir,
    get_chroma_dir
)


class SessionManager:
    """Manages project sessions and lifecycle operations."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize session manager."""
        self.db = db_manager or get_db_manager()

    # ==================== Project Management ====================

    def create_project(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new project with validation.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Dictionary with success status and project data or error message
        """
        # Validate project name
        if not name or not name.strip():
            return {"success": False, "error": "Project name cannot be empty"}

        if len(name) > 255:
            return {"success": False, "error": "Project name is too long (max 255 characters)"}

        # Check project limit
        existing_projects = self.db.get_all_projects()
        if len(existing_projects) >= MAX_PROJECTS_PER_USER:
            return {
                "success": False,
                "error": f"Maximum number of projects ({MAX_PROJECTS_PER_USER}) reached"
            }

        # Check for duplicate name
        for project in existing_projects:
            if project.name.lower() == name.lower():
                return {"success": False, "error": f"Project with name '{name}' already exists"}

        # Create project
        try:
            project = self.db.create_project(name.strip(), description)

            # Create project directories
            get_project_dir(project.id)
            get_uploads_dir(project.id)
            get_web_cache_dir(project.id)
            get_chroma_dir(project.id)

            return {
                "success": True,
                "project": project.to_dict(),
                "message": f"Project '{name}' created successfully"
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create project: {str(e)}"}

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get a project by ID and update last accessed time."""
        project = self.db.get_project(project_id)
        if project:
            self.db.update_project_access(project_id)
        return project

    def get_all_projects(self) -> List[Project]:
        """Get all projects ordered by last accessed."""
        return self.db.get_all_projects()

    def update_project(self, project_id: int, name: Optional[str] = None,
                      description: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a project.

        Args:
            project_id: Project ID
            name: New project name (optional)
            description: New description (optional)

        Returns:
            Dictionary with success status and updated project or error message
        """
        # Validate inputs
        if name is not None:
            if not name.strip():
                return {"success": False, "error": "Project name cannot be empty"}
            if len(name) > 255:
                return {"success": False, "error": "Project name is too long"}

            # Check for duplicate name
            existing_projects = self.db.get_all_projects()
            for project in existing_projects:
                if project.id != project_id and project.name.lower() == name.lower():
                    return {"success": False, "error": f"Project with name '{name}' already exists"}

        try:
            project = self.db.update_project(project_id, name, description)
            if project:
                return {
                    "success": True,
                    "project": project.to_dict(),
                    "message": "Project updated successfully"
                }
            else:
                return {"success": False, "error": "Project not found"}
        except Exception as e:
            return {"success": False, "error": f"Failed to update project: {str(e)}"}

    def delete_project(self, project_id: int) -> Dict[str, Any]:
        """
        Delete a project and all associated data.

        Args:
            project_id: Project ID

        Returns:
            Dictionary with success status and message
        """
        try:
            # Get project first to verify it exists
            project = self.db.get_project(project_id)
            if not project:
                return {"success": False, "error": "Project not found"}

            # Delete database records (cascade will handle documents and conversations)
            success = self.db.delete_project(project_id)

            if success:
                # Delete project directory and all files
                project_dir = get_project_dir(project_id)
                if project_dir.exists():
                    shutil.rmtree(project_dir)

                return {
                    "success": True,
                    "message": f"Project '{project.name}' deleted successfully"
                }
            else:
                return {"success": False, "error": "Failed to delete project from database"}

        except Exception as e:
            return {"success": False, "error": f"Failed to delete project: {str(e)}"}

    def get_project_stats(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a project."""
        project = self.db.get_project(project_id)
        if not project:
            return None

        stats = self.db.get_project_stats(project_id)
        stats["name"] = project.name
        stats["description"] = project.description
        stats["created_at"] = project.created_at.isoformat() if project.created_at else None

        return stats

    # ==================== Document Management ====================

    def can_add_document(self, project_id: int) -> Dict[str, Any]:
        """
        Check if a document can be added to the project.

        Returns:
            Dictionary with can_add boolean and reason if not allowed
        """
        document_count = self.db.get_document_count(project_id)

        if document_count >= MAX_DOCUMENTS_PER_PROJECT:
            return {
                "can_add": False,
                "reason": f"Maximum number of documents ({MAX_DOCUMENTS_PER_PROJECT}) reached for this project"
            }

        return {"can_add": True}

    def add_document(self, project_id: int, source_type: str, source_path: str,
                    title: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Add a document to a project with validation.

        Args:
            project_id: Project ID
            source_type: 'file' or 'url'
            source_path: File path or URL
            title: Document title
            metadata: Additional metadata

        Returns:
            Dictionary with success status and document or error message
        """
        # Check if document can be added
        can_add = self.can_add_document(project_id)
        if not can_add["can_add"]:
            return {"success": False, "error": can_add["reason"]}

        # Validate source type
        if source_type not in ["file", "url"]:
            return {"success": False, "error": "Invalid source type. Must be 'file' or 'url'"}

        try:
            document = self.db.add_document(
                project_id=project_id,
                source_type=source_type,
                source_path=source_path,
                title=title or Path(source_path).name if source_type == "file" else source_path,
                metadata=metadata
            )

            return {
                "success": True,
                "document": document.to_dict(),
                "message": "Document added successfully"
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to add document: {str(e)}"}

    def get_project_documents(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all documents for a project."""
        documents = self.db.get_project_documents(project_id)
        return [doc.to_dict() for doc in documents]

    def delete_document(self, document_id: int) -> Dict[str, Any]:
        """
        Delete a document and its files.

        Args:
            document_id: Document ID

        Returns:
            Dictionary with success status and message
        """
        try:
            # Get document first
            document = self.db.get_document(document_id)
            if not document:
                return {"success": False, "error": "Document not found"}

            # Delete file if it's a file type
            if document.source_type == "file":
                file_path = Path(document.source_path)
                if file_path.exists():
                    file_path.unlink()

            # Delete from database
            success = self.db.delete_document(document_id)

            if success:
                return {
                    "success": True,
                    "message": "Document deleted successfully"
                }
            else:
                return {"success": False, "error": "Failed to delete document"}

        except Exception as e:
            return {"success": False, "error": f"Failed to delete document: {str(e)}"}

    # ==================== Conversation Management ====================

    def add_conversation(self, project_id: int, question: str, answer: str,
                        sources: Optional[List[int]] = None,
                        metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Add a conversation to a project.

        Args:
            project_id: Project ID
            question: User question
            answer: AI answer
            sources: List of document IDs used
            metadata: Additional metadata

        Returns:
            Dictionary with success status and conversation or error message
        """
        try:
            conversation = self.db.add_conversation(
                project_id=project_id,
                question=question,
                answer=answer,
                sources=sources or [],
                metadata=metadata
            )

            # Update project access time
            self.db.update_project_access(project_id)

            return {
                "success": True,
                "conversation": conversation.to_dict(),
                "message": "Conversation saved successfully"
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to save conversation: {str(e)}"}

    def get_project_conversations(self, project_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all conversations for a project."""
        conversations = self.db.get_project_conversations(project_id, limit)
        return [conv.to_dict() for conv in conversations]

    def search_conversations(self, project_id: int, query: str) -> List[Dict[str, Any]]:
        """Search conversations by keyword."""
        conversations = self.db.search_conversations(project_id, query)
        return [conv.to_dict() for conv in conversations]

    def clear_conversations(self, project_id: int) -> Dict[str, Any]:
        """
        Clear all conversations for a project.

        Returns:
            Dictionary with success status and count of deleted conversations
        """
        try:
            count = self.db.clear_project_conversations(project_id)
            return {
                "success": True,
                "deleted_count": count,
                "message": f"Deleted {count} conversation(s)"
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to clear conversations: {str(e)}"}

    # ==================== Utility Methods ====================

    def get_project_size(self, project_id: int) -> Dict[str, Any]:
        """Get storage size information for a project."""
        project_dir = get_project_dir(project_id)

        if not project_dir.exists():
            return {"total_bytes": 0, "total_mb": 0.0}

        total_size = sum(f.stat().st_size for f in project_dir.rglob('*') if f.is_file())

        return {
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2),
            "total_gb": round(total_size / (1024 * 1024 * 1024), 2)
        }

    def cleanup_orphaned_files(self, project_id: int) -> Dict[str, Any]:
        """
        Clean up files that are no longer referenced in the database.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            documents = self.db.get_project_documents(project_id)
            referenced_files = {Path(doc.source_path) for doc in documents if doc.source_type == "file"}

            uploads_dir = get_uploads_dir(project_id)
            deleted_count = 0
            deleted_size = 0

            if uploads_dir.exists():
                for file_path in uploads_dir.rglob('*'):
                    if file_path.is_file() and file_path not in referenced_files:
                        deleted_size += file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1

            return {
                "success": True,
                "deleted_files": deleted_count,
                "freed_bytes": deleted_size,
                "freed_mb": round(deleted_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to cleanup files: {str(e)}"}


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


if __name__ == "__main__":
    # Test session manager
    print("Testing session manager...")

    sm = SessionManager()

    # Create a test project
    result = sm.create_project("Test Session Project", "Testing session management")
    print(f"Create project: {result}")

    if result["success"]:
        project_id = result["project"]["id"]

        # Get project stats
        stats = sm.get_project_stats(project_id)
        print(f"Project stats: {stats}")

        # Add a document
        doc_result = sm.add_document(
            project_id=project_id,
            source_type="url",
            source_path="https://example.com/doc.pdf",
            title="Example Document"
        )
        print(f"Add document: {doc_result}")

        # Add a conversation
        conv_result = sm.add_conversation(
            project_id=project_id,
            question="What is this?",
            answer="This is a test.",
            sources=[doc_result["document"]["id"]] if doc_result["success"] else []
        )
        print(f"Add conversation: {conv_result}")

        # Get updated stats
        stats = sm.get_project_stats(project_id)
        print(f"Updated stats: {stats}")

        # Clean up
        delete_result = sm.delete_project(project_id)
        print(f"Delete project: {delete_result}")

    print("\n✓ Session manager tests completed!")
