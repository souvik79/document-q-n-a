#!/usr/bin/env python3
"""
Clear vector database for a project to allow switching embedding models.

Usage:
    python3 clear_project_vectors.py <project_id>
"""

import sys
import shutil
from pathlib import Path

def clear_project_vectors(project_id: int):
    """Clear vector database for a project."""
    project_dir = Path(f"data/projects/project_{project_id}")

    if not project_dir.exists():
        print(f"❌ Project {project_id} directory not found")
        return False

    chroma_dir = project_dir / "chroma"

    if chroma_dir.exists():
        print(f"📦 Found vector database at: {chroma_dir}")
        print(f"💾 Size: {sum(f.stat().st_size for f in chroma_dir.rglob('*') if f.is_file()) / (1024*1024):.2f} MB")

        response = input(f"\n⚠️  Delete vector database for project {project_id}? (yes/no): ")

        if response.lower() == 'yes':
            shutil.rmtree(chroma_dir)
            print(f"✅ Vector database cleared!")
            print(f"\nℹ️  You can now add documents with the new embedding model (nomic-embed-text)")
            return True
        else:
            print("❌ Cancelled")
            return False
    else:
        print(f"ℹ️  No vector database found for project {project_id}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 clear_project_vectors.py <project_id>")
        print("\nExample:")
        print("  python3 clear_project_vectors.py 2")
        sys.exit(1)

    try:
        project_id = int(sys.argv[1])
        clear_project_vectors(project_id)
    except ValueError:
        print("❌ Invalid project ID. Must be a number.")
        sys.exit(1)
