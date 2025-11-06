"""
Database module for managing SQLite database operations.
Uses SQLAlchemy ORM for database interactions.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

from .config import DATABASE_PATH

# Create declarative base
Base = declarative_base()


class Project(Base):
    """Project/Knowledge Base model."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")

    def to_dict(self, include_counts: bool = False) -> Dict[str, Any]:
        """Convert project to dictionary."""
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

        # Only include counts if explicitly requested and we're in a session
        if include_counts:
            try:
                result["document_count"] = len(self.documents) if self.documents else 0
                result["conversation_count"] = len(self.conversations) if self.conversations else 0
            except:
                pass

        return result


class Document(Base):
    """Document model for files and URLs."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    source_type = Column(String(50), nullable=False)  # 'file' or 'url'
    source_path = Column(Text, nullable=False)  # file path or URL
    title = Column(String(500), nullable=True)
    doc_metadata = Column(JSON, nullable=True)  # JSON: file size, page count, etc.
    status = Column(String(50), default="pending", nullable=False)  # pending, processing, processed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="documents")

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "title": self.title,
            "metadata": self.doc_metadata,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Conversation(Base):
    """Conversation history model."""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # JSON array of document IDs used
    conv_metadata = Column(JSON, nullable=True)  # JSON: model used, tokens, etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    project = relationship("Project", back_populates="conversations")

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.conv_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DatabaseManager:
    """Database manager for handling all database operations."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager."""
        self.db_path = db_path or DATABASE_PATH
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create all tables
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # ==================== Project Operations ====================

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new project."""
        session = self.get_session()
        try:
            project = Project(name=name, description=description)
            session.add(project)
            session.commit()
            session.refresh(project)
            return project
        finally:
            session.close()

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get a project by ID."""
        session = self.get_session()
        try:
            return session.query(Project).filter(Project.id == project_id).first()
        finally:
            session.close()

    def get_all_projects(self) -> List[Project]:
        """Get all projects ordered by last accessed."""
        session = self.get_session()
        try:
            return session.query(Project).order_by(Project.last_accessed.desc()).all()
        finally:
            session.close()

    def update_project(self, project_id: int, name: Optional[str] = None,
                      description: Optional[str] = None) -> Optional[Project]:
        """Update a project."""
        session = self.get_session()
        try:
            project = session.query(Project).filter(Project.id == project_id).first()
            if project:
                if name is not None:
                    project.name = name
                if description is not None:
                    project.description = description
                project.last_accessed = datetime.utcnow()
                session.commit()
                session.refresh(project)
            return project
        finally:
            session.close()

    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all associated data."""
        session = self.get_session()
        try:
            project = session.query(Project).filter(Project.id == project_id).first()
            if project:
                session.delete(project)
                session.commit()
                return True
            return False
        finally:
            session.close()

    def update_project_access(self, project_id: int):
        """Update project's last accessed time."""
        session = self.get_session()
        try:
            project = session.query(Project).filter(Project.id == project_id).first()
            if project:
                project.last_accessed = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    # ==================== Document Operations ====================

    def add_document(self, project_id: int, source_type: str, source_path: str,
                    title: Optional[str] = None, metadata: Optional[Dict] = None) -> Document:
        """Add a new document to a project."""
        session = self.get_session()
        try:
            document = Document(
                project_id=project_id,
                source_type=source_type,
                source_path=source_path,
                title=title,
                metadata=metadata,
                status="pending"
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document
        finally:
            session.close()

    def get_document(self, document_id: int) -> Optional[Document]:
        """Get a document by ID."""
        session = self.get_session()
        try:
            return session.query(Document).filter(Document.id == document_id).first()
        finally:
            session.close()

    def get_project_documents(self, project_id: int) -> List[Document]:
        """Get all documents for a project."""
        session = self.get_session()
        try:
            return session.query(Document).filter(
                Document.project_id == project_id
            ).order_by(Document.created_at.desc()).all()
        finally:
            session.close()

    def update_document_status(self, document_id: int, status: str,
                              error_message: Optional[str] = None) -> Optional[Document]:
        """Update document processing status."""
        session = self.get_session()
        try:
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                document.status = status
                if error_message:
                    document.error_message = error_message
                session.commit()
                session.refresh(document)
            return document
        finally:
            session.close()

    def delete_document(self, document_id: int) -> bool:
        """Delete a document."""
        session = self.get_session()
        try:
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                session.delete(document)
                session.commit()
                return True
            return False
        finally:
            session.close()

    def get_document_count(self, project_id: int) -> int:
        """Get count of documents in a project."""
        session = self.get_session()
        try:
            return session.query(Document).filter(Document.project_id == project_id).count()
        finally:
            session.close()

    # ==================== Conversation Operations ====================

    def add_conversation(self, project_id: int, question: str, answer: str,
                        sources: Optional[List[int]] = None,
                        metadata: Optional[Dict] = None) -> Conversation:
        """Add a new conversation entry."""
        session = self.get_session()
        try:
            conversation = Conversation(
                project_id=project_id,
                question=question,
                answer=answer,
                sources=sources or [],
                metadata=metadata
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation
        finally:
            session.close()

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get a conversation by ID."""
        session = self.get_session()
        try:
            return session.query(Conversation).filter(Conversation.id == conversation_id).first()
        finally:
            session.close()

    def get_project_conversations(self, project_id: int, limit: Optional[int] = None) -> List[Conversation]:
        """Get all conversations for a project."""
        session = self.get_session()
        try:
            query = session.query(Conversation).filter(
                Conversation.project_id == project_id
            ).order_by(Conversation.created_at.asc())

            if limit:
                query = query.limit(limit)

            return query.all()
        finally:
            session.close()

    def get_conversations(self, project_id: int, limit: Optional[int] = None) -> List[Conversation]:
        """Alias for get_project_conversations for backwards compatibility."""
        return self.get_project_conversations(project_id, limit)

    def search_conversations(self, project_id: int, query: str) -> List[Conversation]:
        """Search conversations by keyword."""
        session = self.get_session()
        try:
            search_pattern = f"%{query}%"
            return session.query(Conversation).filter(
                Conversation.project_id == project_id,
                (Conversation.question.like(search_pattern) |
                 Conversation.answer.like(search_pattern))
            ).order_by(Conversation.created_at.desc()).all()
        finally:
            session.close()

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a conversation."""
        session = self.get_session()
        try:
            conversation = session.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            if conversation:
                session.delete(conversation)
                session.commit()
                return True
            return False
        finally:
            session.close()

    def clear_project_conversations(self, project_id: int) -> int:
        """Clear all conversations for a project. Returns count of deleted conversations."""
        session = self.get_session()
        try:
            count = session.query(Conversation).filter(
                Conversation.project_id == project_id
            ).delete()
            session.commit()
            return count
        finally:
            session.close()

    # ==================== Statistics ====================

    def get_project_stats(self, project_id: int) -> Dict[str, Any]:
        """Get statistics for a project."""
        session = self.get_session()
        try:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                return {}

            total_docs = session.query(Document).filter(
                Document.project_id == project_id
            ).count()

            processed_docs = session.query(Document).filter(
                Document.project_id == project_id,
                Document.status == "processed"
            ).count()

            failed_docs = session.query(Document).filter(
                Document.project_id == project_id,
                Document.status == "failed"
            ).count()

            total_conversations = session.query(Conversation).filter(
                Conversation.project_id == project_id
            ).count()

            return {
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "failed_documents": failed_docs,
                "total_conversations": total_conversations,
                "last_accessed": project.last_accessed.isoformat() if project.last_accessed else None,
            }
        finally:
            session.close()


# Global database manager instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


if __name__ == "__main__":
    # Test database operations
    print("Testing database operations...")

    db = DatabaseManager()

    # Create a test project
    project = db.create_project("Test Project", "A test project for Q&A")
    print(f"Created project: {project.to_dict()}")

    # Add a test document
    doc = db.add_document(
        project_id=project.id,
        source_type="file",
        source_path="/path/to/document.pdf",
        title="Test Document",
        metadata={"size": 1024, "pages": 10}
    )
    print(f"Added document: {doc.to_dict()}")

    # Add a test conversation
    conv = db.add_conversation(
        project_id=project.id,
        question="What is this about?",
        answer="This is a test document.",
        sources=[doc.id],
        metadata={"model": "llama3", "tokens": 50}
    )
    print(f"Added conversation: {conv.to_dict()}")

    # Get project stats
    stats = db.get_project_stats(project.id)
    print(f"Project stats: {stats}")

    print("\n✓ Database tests completed successfully!")
