"""
Configuration management for the Document Q&A application.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "document-qna-app")

# Set LangSmith environment variables if API key is provided
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = str(LANGCHAIN_TRACING_V2)
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama3")

# Application Settings
APP_NAME = os.getenv("APP_NAME", "Document Q&A System")
MAX_PROJECTS_PER_USER = int(os.getenv("MAX_PROJECTS_PER_USER", "10"))
MAX_DOCUMENTS_PER_PROJECT = int(os.getenv("MAX_DOCUMENTS_PER_PROJECT", "50"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Text Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Database Configuration
DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "data/app.db")

# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY = BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/projects")

# Search Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")  # similarity or mmr

# LLM Parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# Cloud Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # 'local' or 'gemini'
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "local")  # 'local' or 'gemini'

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

# Supported file types
SUPPORTED_FILE_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
    # Image formats
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "webp": "image/webp",
    "gif": "image/gif",
}

# Export formats
EXPORT_FORMATS = ["json", "csv", "pdf"]

def get_project_dir(project_id: int) -> Path:
    """Get the directory path for a specific project."""
    project_path = PROJECTS_DIR / f"project_{project_id}"
    project_path.mkdir(exist_ok=True)
    return project_path

def get_uploads_dir(project_id: int) -> Path:
    """Get the uploads directory for a specific project."""
    uploads_path = get_project_dir(project_id) / "uploads"
    uploads_path.mkdir(exist_ok=True)
    return uploads_path

def get_web_cache_dir(project_id: int) -> Path:
    """Get the web cache directory for a specific project."""
    web_cache_path = get_project_dir(project_id) / "web_cache"
    web_cache_path.mkdir(exist_ok=True)
    return web_cache_path

def get_chroma_dir(project_id: int) -> Path:
    """Get the ChromaDB directory for a specific project."""
    chroma_path = get_project_dir(project_id) / "chroma_db"
    chroma_path.mkdir(exist_ok=True)
    return chroma_path

def validate_config() -> dict:
    """
    Validate configuration and return status.
    Returns dict with 'valid' boolean and 'errors' list.
    """
    errors = []

    # Check if LangSmith API key is set
    if not LANGCHAIN_API_KEY:
        errors.append("LANGCHAIN_API_KEY not set. LangSmith tracing will be disabled.")

    # Check if Ollama is accessible
    try:
        import requests
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            errors.append(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    except Exception as e:
        errors.append(f"Ollama connection error: {str(e)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": [e for e in errors if "LangSmith" in e]
    }

if __name__ == "__main__":
    # Print configuration for debugging
    print("=" * 50)
    print(f"Configuration for {APP_NAME}")
    print("=" * 50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Database Path: {DATABASE_PATH}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print(f"LangSmith Tracing: {LANGCHAIN_TRACING_V2}")
    print(f"LangSmith Project: {LANGCHAIN_PROJECT}")
    print("=" * 50)

    # Validate configuration
    validation = validate_config()
    if validation["valid"]:
        print("✓ Configuration is valid!")
    else:
        print("⚠ Configuration issues found:")
        for error in validation["errors"]:
            print(f"  - {error}")
