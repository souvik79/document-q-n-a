# 🤖 Document Q&A System

> **Intelligent Document Question-Answering with Multi-Modal AI**
> 
> Chat with your documents (PDF, DOCX, Images), leverage powerful RAG, and toggle between local (Ollama) and cloud (Gemini) AI models.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B.svg)](https://streamlit.io/)

---

## ✨ Key Features

### 📚 Multi-Format Document Support
- **Text Documents**: PDF, DOCX, TXT, Markdown, CSV
- **Images**: PNG, JPG, JPEG, BMP, TIFF, WEBP (with OCR + Vision AI)
- **Web Content**: Scrape and index URLs
- **Multiple Sources**: Combine different document types in one knowledge base

### 🧠 Flexible AI Providers
- **Local (Ollama)**: 100% private, free, runs on your machine
- **Cloud (Gemini)**: 5-10x faster, free tier available (60 req/min)
- **Easy Toggle**: Switch between providers with one setting
- **Hybrid Mode**: Use different providers for different tasks

### 🎯 Powerful Features
- **Multi-Project Management**: Create isolated knowledge bases
- **Persistent Conversations**: Full chat history with search
- **Source Attribution**: See which documents answered each question
- **Image Understanding**: OCR text extraction + AI vision analysis
- **Export Capabilities**: JSON, CSV, PDF export formats
- **Improved Retrieval**: Hybrid search (semantic + keyword matching)

### 🚀 Modern Interface
- **Streamlit Web UI**: Beautiful, intuitive interface
- **Project Dashboard**: Manage multiple knowledge bases
- **Real-time Progress**: See document processing status
- **History & Search**: Find past conversations instantly

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: For local AI models (optional if using Gemini)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space

### Installation (5 Minutes)

#### 1. Install Ollama (Optional - for local models)

```bash
# Linux/WSL
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Start Ollama
ollama serve

# Pull models (in another terminal)
ollama pull qwen2.5:14b           # Q&A model
ollama pull nomic-embed-text      # Embeddings
ollama pull llava:7b              # Vision (optional)
```

#### 2. Clone and Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/document_q_n_a.git
cd document_q_n_a

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys (optional)
```

#### 3. Run the Application

```bash
# Activate environment
source env/bin/activate

# Start the app
streamlit run app.py

# Open browser to: http://localhost:8501
```

**That's it!** 🎉

---

## ⚙️ Configuration

### Quick Setup (Local Only)

For local-only setup, no API keys needed. Just ensure Ollama is running:

```env
LLM_PROVIDER=local
VISION_PROVIDER=local
```

### Cloud Setup (Faster Responses)

Get 5-10x faster responses with Google Gemini (free tier):

1. Get API key: https://makersuite.google.com/app/apikey
2. Update `.env`:

```env
LLM_PROVIDER=gemini
VISION_PROVIDER=gemini
GOOGLE_API_KEY=your_api_key_here
```

### Hybrid Setup (Best of Both)

```env
LLM_PROVIDER=gemini    # Fast Q&A
VISION_PROVIDER=local  # Private image processing
```

### Full Configuration Options

See [.env.example](.env.example) for all available settings including:
- Model selection
- Chunk sizes
- Rate limits
- Storage paths
- LangSmith tracking (optional)

---

## 📖 Usage Guide

### Creating Your First Knowledge Base

1. **Create Project**
   - Go to "Knowledge Base" tab
   - Click "Create New Project"
   - Name it and add description

2. **Add Documents**
   - Upload files (PDF, DOCX, images, etc.)
   - Or add URLs to scrape
   - Wait for processing (see progress bar)

3. **Start Chatting**
   - Go to "Chat" tab
   - Ask questions about your documents
   - See source attribution for each answer

4. **Review History**
   - Go to "History" tab
   - Search past conversations
   - Export to JSON/CSV/PDF

### Image Processing

The system can extract text from images AND understand visual content:

```
Upload image.png → 
  ├─ OCR (Tesseract) → Extract text
  ├─ Vision AI → Describe image content
  └─ Combined → Full understanding
```

**Supported formats**: PNG, JPG, JPEG, BMP, TIFF, WEBP, GIF

### Provider Comparison

| Feature | Local (Ollama) | Cloud (Gemini) |
|---------|----------------|----------------|
| **Speed** | 20-45 sec/query | 2-5 sec/query |
| **Cost** | Free forever | Free tier (60/min) |
| **Privacy** | 100% private | Data sent to Google |
| **Internet** | Not required | Required |
| **Quality** | Excellent | Excellent |
| **Images** | LLaVA (slow) | Gemini Vision (fast) |

---

## 🏗️ Architecture

### Tech Stack

- **LLM Framework**: LangChain
- **Local Models**: Ollama (Qwen 2.5, LLaVA)
- **Cloud Models**: Google Gemini 2.5 Flash
- **Vector DB**: ChromaDB
- **Database**: SQLite (SQLAlchemy ORM)
- **UI**: Streamlit
- **OCR**: Tesseract, EasyOCR
- **Document Processing**: PyPDF, python-docx, BeautifulSoup

### How RAG Works

```
┌──────────────┐
│ User Question│
└──────┬───────┘
       │
       v
┌──────────────────────┐
│ Hybrid Search        │  1. Semantic search (vector similarity)
│ (Semantic + Keyword) │  2. Keyword matching (BM25)
│                      │  3. Merge & re-rank results
└──────┬───────────────┘
       │
       v
┌──────────────────┐
│ Retrieved Chunks │  Top 10 most relevant passages
└──────┬───────────┘
       │
       v
┌──────────────────┐
│ LLM Generation   │  Qwen 2.5 or Gemini
│ (with context)   │  → Grounded answer
└──────┬───────────┘
       │
       v
┌──────────────────┐
│ Answer + Sources │  With document attribution
└──────────────────┘
```

### Project Structure

```
document_q_n_a/
├── app.py                      # Streamlit UI
├── src/
│   ├── config.py               # Configuration management
│   ├── database.py             # SQLAlchemy models
│   ├── cloud_providers.py      # Gemini API wrapper
│   ├── document_processor.py   # Document parsing
│   ├── image_processor.py      # OCR + Vision AI
│   ├── vector_store.py         # ChromaDB operations
│   ├── qa_chain.py             # RAG implementation
│   ├── improved_retrieval.py   # Hybrid search
│   ├── web_scraper.py          # URL fetching
│   ├── export_manager.py       # Export functionality
│   └── session_manager.py      # Project management
├── data/                       # Application data
│   ├── app.db                  # SQLite database
│   └── projects/               # Project vector stores
├── docs_backup/                # Documentation backups
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
└── README.md                   # This file
```

---

## 🎯 Performance

### Response Times

| Task | Local (Ollama) | Cloud (Gemini) |
|------|----------------|----------------|
| Q&A | 20-45 seconds | 2-5 seconds |
| Image Vision | 3-12 seconds | 1-3 seconds |
| OCR | 0.5-2 seconds | 0.5-2 seconds |
| Document Upload | 1-5 seconds | 1-5 seconds |

### Resource Usage

- **CPU**: Qwen 2.5:14b uses all available cores
- **RAM**: ~6-8GB for model + application
- **Disk**: ~5GB for models + your documents
- **GPU**: Optional (3-5x speedup if available)

---

## 🔧 Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Verify models are pulled
ollama list
```

### Slow Performance

- **Use Gemini** for 5-10x faster responses (set `LLM_PROVIDER=gemini`)
- **Use GPU** if available (Ollama auto-detects)
- **Reduce model size**: Try `qwen2.5:7b` instead of `14b`

### Image Processing Not Working

```bash
# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### "No module named 'src'"

```bash
# Make sure you're in the project directory
cd document_q_n_a

# Activate virtual environment
source env/bin/activate
```

---

## 🛣️ Roadmap

### ✅ Completed (Current Version)
- Multi-format document support
- Image processing (OCR + Vision)
- Cloud provider integration (Gemini)
- Hybrid search (semantic + keyword)
- Project management
- Conversation history
- Export functionality

### 🚧 Planned Features
- **Phase 1** (Next 2-4 weeks):
  - Text-to-Image generation (Stable Diffusion)
  - Speech-to-Text (Whisper)
  - Text-to-Speech (Coqui TTS)
  
- **Phase 2** (1-2 months):
  - Video analysis
  - Audio processing
  - Code assistant
  - Advanced analytics dashboard

- **Phase 3** (Future):
  - Multi-user support
  - User authentication
  - API endpoints
  - Mobile app

See [AI_SERVICE_EXPANSION_PLAN.md](docs_backup/AI_SERVICE_EXPANSION_PLAN.md) for details.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Feel free to use for personal or commercial projects!

---

## 🙏 Acknowledgments

Built with these amazing open-source projects:

- [Ollama](https://ollama.com/) - Local LLM runtime
- [LangChain](https://python.langchain.com/) - LLM orchestration framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web UI framework
- [Google Gemini](https://ai.google.dev/) - Cloud AI API
- [Qwen 2.5](https://qwenlm.github.io/) by Alibaba Cloud
- [LLaVA](https://llava-vl.github.io/) - Vision-Language model

---

## 📞 Support

- **Documentation**: See [docs_backup/](docs_backup/) for detailed guides
- **Issues**: Open an issue on GitHub
- **Questions**: Check documentation or open a discussion

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Ready to start?** Follow the [Quick Start](#-quick-start) guide above! 🚀
