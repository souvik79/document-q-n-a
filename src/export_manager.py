"""
Export Manager for Document Q&A System.

Handles exporting conversations and project data to various formats:
- JSON: Structured data export
- CSV: Tabular data for spreadsheets
- PDF: Formatted report with conversations
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import StringIO, BytesIO

from fpdf import FPDF
from src.database import get_db_manager
from src.session_manager import SessionManager


class ConversationPDF(FPDF):
    """Custom PDF class for conversation export."""

    def __init__(self, project_name: str):
        super().__init__()
        self.project_name = project_name

    def header(self):
        """PDF header with project name."""
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, f'Conversation History: {self.project_name}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """PDF footer with page number."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title: str):
        """Add a conversation title."""
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(2)

    def chapter_body(self, question: str, answer: str, metadata: Optional[Dict] = None):
        """Add conversation content."""
        # Question
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, 'Question:', 0, 1)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, question)
        self.ln(2)

        # Answer
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, 'Answer:', 0, 1)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, answer)
        self.ln(2)

        # Metadata
        if metadata:
            self.set_font('Arial', 'I', 9)
            self.set_text_color(128, 128, 128)

            meta_text = f"Query time: {metadata.get('query_time_seconds', 'N/A')}s | "
            meta_text += f"Sources: {metadata.get('num_sources', 'N/A')}"

            self.cell(0, 5, meta_text, 0, 1)
            self.set_text_color(0, 0, 0)

        self.ln(5)


class ExportManager:
    """Manages data export for conversations and projects."""

    def __init__(self, project_id: int):
        """
        Initialize export manager.

        Args:
            project_id: The project to export data from
        """
        self.project_id = project_id
        self.db = get_db_manager()
        self.sm = SessionManager()

        # Get project info
        self.project = self.sm.get_project(project_id)
        if not self.project:
            raise ValueError(f"Project {project_id} not found")

    def export_to_json(self, include_documents: bool = True, include_stats: bool = True) -> str:
        """
        Export project data to JSON format.

        Args:
            include_documents: Include document list
            include_stats: Include project statistics

        Returns:
            JSON string with project data
        """
        data = {
            "project": {
                "id": self.project.id,
                "name": self.project.name,
                "description": self.project.description,
                "created_at": self.project.created_at.isoformat(),
            },
            "export_date": datetime.now().isoformat(),
        }

        # Add statistics
        if include_stats:
            stats = self.sm.get_project_stats(self.project_id)
            size = self.sm.get_project_size(self.project_id)
            data["statistics"] = {
                **stats,
                "storage_mb": size["total_mb"]
            }

        # Add documents
        if include_documents:
            documents = self.sm.get_project_documents(self.project_id)
            data["documents"] = [
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "type": doc["source_type"],
                    "source": doc["source_path"],
                    "status": doc["status"],
                    "created_at": doc["created_at"].isoformat(),
                }
                for doc in documents
            ]

        # Add conversations
        conversations = self.db.get_conversations(self.project_id)
        data["conversations"] = [
            {
                "id": conv.id,
                "question": conv.question,
                "answer": conv.answer,
                "created_at": conv.created_at.isoformat(),
                "metadata": conv.conv_metadata,
                "source_documents": conv.source_documents,
            }
            for conv in conversations
        ]

        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_to_csv(self) -> str:
        """
        Export conversations to CSV format.

        Returns:
            CSV string with conversations
        """
        conversations = self.db.get_conversations(self.project_id)

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "ID",
            "Date",
            "Question",
            "Answer",
            "Query Time (s)",
            "Sources Used",
        ])

        # Data
        for conv in conversations:
            writer.writerow([
                conv.id,
                conv.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                conv.question,
                conv.answer,
                conv.conv_metadata.get("query_time_seconds", "N/A") if conv.conv_metadata else "N/A",
                conv.conv_metadata.get("num_sources", "N/A") if conv.conv_metadata else "N/A",
            ])

        return output.getvalue()

    def export_to_pdf(self) -> bytes:
        """
        Export conversations to PDF format.

        Returns:
            PDF file as bytes
        """
        conversations = self.db.get_conversations(self.project_id)

        # Create PDF
        pdf = ConversationPDF(self.project.name)
        pdf.add_page()

        # Project info
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f'Project: {self.project.name}', 0, 1)
        if self.project.description:
            pdf.cell(0, 6, f'Description: {self.project.description}', 0, 1)
        pdf.cell(0, 6, f'Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        pdf.cell(0, 6, f'Total Conversations: {len(conversations)}', 0, 1)
        pdf.ln(10)

        # Conversations
        for i, conv in enumerate(conversations, 1):
            # Add new page if needed
            if pdf.get_y() > 250:
                pdf.add_page()

            # Add conversation
            title = f"Conversation {i} - {conv.created_at.strftime('%Y-%m-%d %H:%M')}"
            pdf.chapter_title(title)
            pdf.chapter_body(
                conv.question,
                conv.answer,
                conv.conv_metadata
            )

        # Return PDF as bytes
        return pdf.output(dest='S').encode('latin-1')

    def save_to_file(self, format: str, filepath: Path) -> Dict[str, Any]:
        """
        Export and save to file.

        Args:
            format: Export format ('json', 'csv', 'pdf')
            filepath: Where to save the file

        Returns:
            Result dictionary with success status
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                content = self.export_to_json()
                filepath.write_text(content, encoding='utf-8')
            elif format == "csv":
                content = self.export_to_csv()
                filepath.write_text(content, encoding='utf-8')
            elif format == "pdf":
                content = self.export_to_pdf()
                filepath.write_bytes(content)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}"
                }

            return {
                "success": True,
                "message": f"Exported to {filepath}",
                "filepath": str(filepath)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Export failed: {str(e)}"
            }


def export_all_projects(output_dir: Path) -> Dict[str, Any]:
    """
    Export all projects to separate files.

    Args:
        output_dir: Directory to save exports

    Returns:
        Result dictionary with export summary
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sm = SessionManager()
        projects = sm.get_all_projects()

        if not projects:
            return {
                "success": False,
                "error": "No projects to export"
            }

        results = []

        for project in projects:
            try:
                exporter = ExportManager(project.id)

                # Create project subdirectory
                project_dir = output_dir / f"project_{project.id}_{project.name.replace(' ', '_')}"
                project_dir.mkdir(exist_ok=True)

                # Export to all formats
                for format in ["json", "csv", "pdf"]:
                    filename = f"{project.name.replace(' ', '_')}_conversations.{format}"
                    filepath = project_dir / filename

                    result = exporter.save_to_file(format, filepath)
                    results.append({
                        "project": project.name,
                        "format": format,
                        **result
                    })

            except Exception as e:
                results.append({
                    "project": project.name,
                    "success": False,
                    "error": str(e)
                })

        success_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)

        return {
            "success": True,
            "message": f"Exported {success_count}/{total_count} files successfully",
            "results": results,
            "output_dir": str(output_dir)
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Export failed: {str(e)}"
        }
