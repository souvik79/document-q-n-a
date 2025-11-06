"""
Streamlit UI for Document Q&A System.

A modern, user-friendly interface for managing projects, uploading documents,
and chatting with your knowledge base.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.session_manager import SessionManager
from src.document_processor import DocumentProcessor
from src.web_scraper import WebScraper
from src.vector_store import VectorStoreManager
from src.qa_chain import QAChain
from src.database import get_db_manager
from src.export_manager import ExportManager

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()

if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing" not in st.session_state:
    st.session_state.processing = False


def format_timestamp(timestamp) -> str:
    """Format timestamp for display."""
    # Handle both datetime objects and strings
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            return timestamp  # Return as-is if can't parse

    if not isinstance(timestamp, datetime):
        return str(timestamp)

    now = datetime.now()

    # Make timestamp timezone-naive if it's timezone-aware
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    diff = now - timestamp

    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"
    else:
        return "just now"


def render_sidebar():
    """Render the sidebar with project selection and navigation."""
    with st.sidebar:
        st.markdown("### 📚 Document Q&A System")
        st.markdown("---")

        # Project selection
        st.markdown("#### Current Project")

        sm = st.session_state.session_manager
        projects = sm.get_all_projects()

        if projects:
            project_options = {
                f"{p.name} (ID: {p.id})": p.id
                for p in projects
            }

            # Find current selection
            current_name = None
            if st.session_state.current_project_id:
                for name, pid in project_options.items():
                    if pid == st.session_state.current_project_id:
                        current_name = name
                        break

            selected = st.selectbox(
                "Select a project",
                options=list(project_options.keys()),
                index=list(project_options.values()).index(st.session_state.current_project_id)
                      if st.session_state.current_project_id and st.session_state.current_project_id in project_options.values()
                      else 0,
                key="project_selector"
            )

            st.session_state.current_project_id = project_options[selected]

            # Project stats
            if st.session_state.current_project_id:
                stats = sm.get_project_stats(st.session_state.current_project_id)
                size = sm.get_project_size(st.session_state.current_project_id)

                st.markdown("#### Project Stats")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents", stats.get('total_documents', 0))
                with col2:
                    st.metric("Conversations", stats.get('total_conversations', 0))

                st.metric("Storage", f"{size['total_mb']:.2f} MB")
        else:
            st.info("No projects yet. Create one in the Projects tab!")

        st.markdown("---")

        # Navigation
        st.markdown("#### Navigation")
        page = st.radio(
            "Go to",
            ["Projects", "Knowledge Base", "Chat", "History", "Export"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System info
        st.markdown("#### System")
        st.caption("Local LLM via Ollama")
        st.caption("Vector DB: ChromaDB")

        return page


def render_projects_page():
    """Render the projects management page."""
    st.markdown('<p class="main-header">Project Management</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Create and manage your knowledge bases</p>', unsafe_allow_html=True)

    sm = st.session_state.session_manager

    # Create new project
    st.markdown("### Create New Project")

    col1, col2 = st.columns([2, 1])

    with col1:
        new_name = st.text_input("Project Name", placeholder="My Research Project")
        new_desc = st.text_area("Description (optional)", placeholder="Describe your project...")

    with col2:
        st.markdown("###")
        if st.button("Create Project", type="primary"):
            if new_name:
                result = sm.create_project(new_name, new_desc if new_desc else None)

                if result["success"]:
                    st.session_state.current_project_id = result["project"]["id"]
                    st.success(f"✅ Project '{new_name}' created successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ {result['error']}")
            else:
                st.warning("Please enter a project name")

    st.markdown("---")

    # List existing projects
    st.markdown("### Your Projects")

    projects = sm.get_all_projects()

    if projects:
        for project in projects:
            with st.expander(
                f"{'🔵 ' if project.id == st.session_state.current_project_id else ''}**{project.name}** (ID: {project.id})",
                expanded=project.id == st.session_state.current_project_id
            ):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.markdown(f"**Description:** {project.description or 'No description'}")
                    st.caption(f"Created: {format_timestamp(project.created_at)}")

                with col2:
                    stats = sm.get_project_stats(project.id)
                    st.markdown(f"**Documents:** {stats.get('total_documents', 0)}")
                    st.markdown(f"**Conversations:** {stats.get('total_conversations', 0)}")

                with col3:
                    if st.button("Select", key=f"select_{project.id}"):
                        st.session_state.current_project_id = project.id
                        st.rerun()

                    if st.button("Delete", key=f"delete_{project.id}"):
                        if st.session_state.current_project_id == project.id:
                            st.session_state.current_project_id = None

                        result = sm.delete_project(project.id)
                        if result["success"]:
                            st.success("Project deleted!")
                            st.rerun()
                        else:
                            st.error(result["error"])
    else:
        st.info("No projects yet. Create one above to get started!")


def render_knowledge_base_page():
    """Render the knowledge base management page."""
    if not st.session_state.current_project_id:
        st.warning("Please select a project first from the sidebar")
        return

    st.markdown('<p class="main-header">Knowledge Base</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Add and manage documents in your project</p>', unsafe_allow_html=True)

    sm = st.session_state.session_manager
    project_id = st.session_state.current_project_id

    # Tabs for different upload methods
    tab1, tab2, tab3 = st.tabs(["📄 Upload File", "🌐 Add URL", "📚 View Documents"])

    with tab1:
        st.markdown("### Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "gif"],
            help="Supported formats: PDF, DOCX, TXT, MD, CSV, PNG, JPG, JPEG, BMP, TIFF, WEBP, GIF"
        )

        if uploaded_file:
            # Show image preview if it's an image file
            file_extension = Path(uploaded_file.name).suffix.lower().lstrip('.')
            if file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'gif']:
                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
                st.info("📸 Image will be processed with OCR and vision model to extract text and generate description")

            if st.button("Process File", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = Path("data/temp") / uploaded_file.name
                        temp_path.parent.mkdir(exist_ok=True)
                        temp_path.write_bytes(uploaded_file.read())

                        # Process file
                        processor = DocumentProcessor()

                        # Validate
                        validation = processor.validate_file(temp_path)
                        if not validation["valid"]:
                            st.error(f"❌ {validation['error']}")
                            temp_path.unlink()
                            return

                        # Create chunks
                        file_ext = Path(uploaded_file.name).suffix.lower().lstrip('.')
                        if file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'gif']:
                            st.info("📸 Processing image...")
                            st.info("   🔍 Extracting text with OCR...")
                            st.info("   👁️ Analyzing image with vision model...")
                        else:
                            st.info("Creating chunks...")

                        chunks, metadata = processor.process_file(temp_path)
                        st.success(f"✓ Created {len(chunks)} chunks")

                        # Add to database
                        doc_result = sm.add_document(
                            project_id=project_id,
                            source_type="file",
                            source_path=str(temp_path),
                            title=uploaded_file.name,
                            metadata=metadata
                        )

                        if not doc_result["success"]:
                            st.error(f"❌ {doc_result['error']}")
                            temp_path.unlink()
                            return

                        document_id = doc_result["document"]["id"]

                        # Add to vector store
                        st.info("Adding to vector database...")
                        vs = VectorStoreManager(project_id)
                        vs_result = vs.add_documents(chunks, document_id=document_id)

                        if vs_result["success"]:
                            db = get_db_manager()
                            db.update_document_status(document_id, "processed")
                            st.success(f"✅ Document added successfully! (ID: {document_id})")
                        else:
                            st.error(f"❌ {vs_result['error']}")

                        # Clean up
                        temp_path.unlink()

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with tab2:
        st.markdown("### Add URL")

        url = st.text_input("Enter URL", placeholder="https://example.com/article")

        if url:
            if st.button("Process URL", type="primary"):
                with st.spinner("Processing URL..."):
                    try:
                        scraper = WebScraper()

                        # Validate
                        validation = scraper.validate_url(url)
                        if not validation["valid"]:
                            st.error(f"❌ {validation['error']}")
                            return

                        # Fetch and process
                        st.info("Fetching content...")
                        chunks, metadata = scraper.process_url(url, project_id)
                        st.success(f"✓ Created {len(chunks)} chunks")

                        # Add to database
                        doc_result = sm.add_document(
                            project_id=project_id,
                            source_type="url",
                            source_path=url,
                            title=metadata.get("title", url),
                            metadata=metadata
                        )

                        if not doc_result["success"]:
                            st.error(f"❌ {doc_result['error']}")
                            return

                        document_id = doc_result["document"]["id"]

                        # Add to vector store
                        st.info("Adding to vector database...")
                        vs = VectorStoreManager(project_id)
                        vs_result = vs.add_documents(chunks, document_id=document_id)

                        if vs_result["success"]:
                            db = get_db_manager()
                            db.update_document_status(document_id, "processed")
                            st.success(f"✅ URL added successfully! (ID: {document_id})")
                            st.info(f"Title: {metadata.get('title', 'N/A')}")
                        else:
                            st.error(f"❌ {vs_result['error']}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with tab3:
        st.markdown("### Documents in Project")

        documents = sm.get_project_documents(project_id)

        if documents:
            for doc in documents:
                with st.expander(f"**{doc['title']}** (ID: {doc['id']})"):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Type:** {doc['source_type']}")
                        st.markdown(f"**Status:** {doc['status']}")

                        if doc['source_type'] == 'url':
                            st.markdown(f"**URL:** {doc['source_path']}")
                        else:
                            st.markdown(f"**File:** {Path(doc['source_path']).name}")

                        st.caption(f"Added: {format_timestamp(doc['created_at'])}")

                    with col2:
                        if st.button("Delete", key=f"doc_delete_{doc['id']}"):
                            # Delete from vector store
                            vs = VectorStoreManager(project_id)
                            vs.delete_by_document_id(doc['id'])

                            # Delete from database
                            db = get_db_manager()
                            db.delete_document(doc['id'])

                            st.success("Document deleted!")
                            st.rerun()
        else:
            st.info("No documents in this project yet. Add some using the tabs above!")

        # Compact database button (always show)
        st.markdown("---")
        st.markdown("### Database Maintenance")
        st.caption("Compact the database to reclaim disk space after deleting documents")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗜️ Compact Database", help="Rebuild database to reclaim space from deleted documents"):
                with st.spinner("Compacting database..."):
                    vs = VectorStoreManager(project_id)
                    result = vs.compact_database()

                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.info(f"Space saved: {result.get('space_saved_mb', 0):.2f} MB")
                        if result.get('chunks_kept'):
                            st.caption(f"Kept {result['chunks_kept']} chunks")
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('error', 'Unknown error')}")

        with col2:
            if st.button("🗑️ Clear Vector DB", help="Clear vector database to switch embedding models", type="secondary"):
                if st.session_state.get('confirm_clear_vectors'):
                    with st.spinner("Clearing vector database..."):
                        import shutil
                        chroma_dir = Path(f"data/projects/project_{project_id}/chroma")
                        if chroma_dir.exists():
                            shutil.rmtree(chroma_dir)
                            st.success("✅ Vector database cleared! You can now add documents with the new embedding model.")
                            st.session_state.confirm_clear_vectors = False
                            st.rerun()
                        else:
                            st.info("No vector database to clear")
                            st.session_state.confirm_clear_vectors = False
                else:
                    st.warning("⚠️ This will delete all vector embeddings! Click again to confirm.")
                    st.session_state.confirm_clear_vectors = True


def render_chat_page():
    """Render the chat interface page."""
    if not st.session_state.current_project_id:
        st.warning("Please select a project first from the sidebar")
        return

    st.markdown('<p class="main-header">Chat with Your Documents</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your knowledge base</p>', unsafe_allow_html=True)

    sm = st.session_state.session_manager
    project_id = st.session_state.current_project_id

    # Check if documents exist
    documents = sm.get_project_documents(project_id)

    if not documents:
        st.warning("No documents in this project yet. Add some in the Knowledge Base tab first!")
        return

    # Display chat history
    st.markdown("### Conversation")

    # Add option to show sources
    show_sources = st.checkbox("Show source context", value=False, key="show_sources_toggle")

    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 <strong>Assistant:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

            # Show metadata if available
            if "metadata" in message:
                metadata = message["metadata"]
                col1, col2, col3 = st.columns([2, 2, 6])
                with col1:
                    st.caption(f"⏱️ {metadata.get('query_time_seconds', 0)}s")
                with col2:
                    st.caption(f"📚 {metadata.get('num_sources', 0)} sources")
                with col3:
                    # Answer quality feedback
                    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 8])
                    with feedback_col1:
                        if st.button("👍", key=f"upvote_{i}", help="Good answer"):
                            st.session_state.chat_history[i]["feedback"] = "positive"
                            st.success("Thank you for your feedback!")
                    with feedback_col2:
                        if st.button("👎", key=f"downvote_{i}", help="Poor answer"):
                            st.session_state.chat_history[i]["feedback"] = "negative"
                            st.warning("Thank you for your feedback. We'll work on improving!")

            # Show sources if available and toggle is on
            if show_sources and "sources" in message:
                with st.expander(f"📄 View Sources ({len(message['sources'])} chunks used)"):
                    for j, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {j}:**")
                        st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                        st.caption(f"Metadata: {source.get('metadata', {})}")
                        st.markdown("---")

    # Question input
    st.markdown("---")

    # Show suggested questions if chat is empty
    if not st.session_state.chat_history:
        st.markdown("### Suggested Questions")
        st.caption("Click a suggestion to ask that question")

        suggestions = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key takeaways?",
            "What endpoints are available?",
        ]

        cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                    st.session_state.suggested_question = suggestion
                    st.rerun()

        st.markdown("---")

    # Check if we have a suggested question to process
    suggested_question = st.session_state.get('suggested_question', '')
    if suggested_question:
        # Clear the suggestion
        st.session_state.suggested_question = ''
        question = suggested_question
        should_ask = True
    else:
        should_ask = False

    col1, col2 = st.columns([5, 1])

    with col1:
        question_input = st.text_input(
            "Ask a question",
            placeholder="What is this document about?",
            key="question_input",
            value=question if should_ask else "",
            label_visibility="collapsed"
        )

        if not should_ask:
            question = question_input

    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    # Process question if suggested or button clicked
    should_ask = should_ask or ask_button

    if should_ask and question:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Get answer with detailed progress
        progress_text = st.empty()
        progress_bar = st.progress(0)

        try:
            # Step 1: Initialize
            progress_text.text("🔍 Searching documents...")
            progress_bar.progress(25)

            qa = QAChain(project_id)

            # Step 2: Retrieve
            progress_text.text("📚 Retrieving relevant context...")
            progress_bar.progress(50)

            result = qa.ask(question)

            # Step 3: Generate
            progress_text.text("💭 Generating answer...")
            progress_bar.progress(75)

            if result["success"]:
                answer = result["answer"]

                # Add assistant message to history with sources
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": result.get("sources", []),
                    "metadata": result.get("metadata", {})
                })

                # Save conversation
                sm.add_conversation(
                    project_id=project_id,
                    question=question,
                    answer=answer,
                    sources=result.get("source_document_ids", []),
                    metadata=result["metadata"]
                )

                # Complete
                progress_text.text("✅ Done!")
                progress_bar.progress(100)

                st.rerun()
            else:
                progress_text.empty()
                progress_bar.empty()
                st.error(f"❌ {result['error']}")

        except Exception as e:
            progress_text.empty()
            progress_bar.empty()
            st.error(f"❌ Error: {str(e)}")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def render_history_page():
    """Render the conversation history page."""
    if not st.session_state.current_project_id:
        st.warning("Please select a project first from the sidebar")
        return

    st.markdown('<p class="main-header">Conversation History</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View and search past conversations</p>', unsafe_allow_html=True)

    sm = st.session_state.session_manager
    project_id = st.session_state.current_project_id

    # Get conversations
    db = get_db_manager()
    conversations = db.get_conversations(project_id)

    if conversations:
        # Search functionality
        st.markdown("### Search Conversations")

        col1, col2 = st.columns([4, 1])

        with col1:
            search_query = st.text_input(
                "Search in questions and answers",
                placeholder="Enter keywords to search...",
                key="search_conversations"
            )

        with col2:
            st.markdown("###")
            search_button = st.button("Search", type="primary", use_container_width=True)

        # Filter conversations based on search
        filtered_conversations = conversations

        if search_query:
            search_lower = search_query.lower()
            filtered_conversations = [
                conv for conv in conversations
                if search_lower in conv.question.lower() or search_lower in conv.answer.lower()
            ]

        st.markdown("---")

        # Display results
        if filtered_conversations:
            st.markdown(f"### {'Search Results' if search_query else 'Recent Conversations'} ({len(filtered_conversations)})")

            # Limit display to 20 most recent
            display_conversations = filtered_conversations[:20]

            if len(filtered_conversations) > 20:
                st.info(f"Showing 20 most recent of {len(filtered_conversations)} results")

            for conv in display_conversations:
                with st.expander(f"**Q:** {conv.question[:100]}... - {format_timestamp(conv.created_at)}"):
                    st.markdown(f"**Question:**\n{conv.question}")
                    st.markdown("---")
                    st.markdown(f"**Answer:**\n{conv.answer}")

                    # Show metadata if available
                    if conv.conv_metadata:
                        st.caption(f"Query time: {conv.conv_metadata.get('query_time_seconds', 'N/A')}s")
                        st.caption(f"Sources used: {conv.conv_metadata.get('num_sources', 'N/A')}")
        else:
            st.warning(f"No conversations found matching '{search_query}'")
    else:
        st.info("No conversations yet. Start chatting in the Chat tab!")


def render_export_page():
    """Render the export page."""
    if not st.session_state.current_project_id:
        st.warning("Please select a project first from the sidebar")
        return

    st.markdown('<p class="main-header">Export Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Export conversations and project data</p>', unsafe_allow_html=True)

    sm = st.session_state.session_manager
    project_id = st.session_state.current_project_id

    # Get project info
    project = sm.get_project(project_id)
    db = get_db_manager()
    conversations = db.get_conversations(project_id)

    st.markdown("### Export Options")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info(f"**Project:** {project.name}\n\n**Conversations:** {len(conversations)}")

    with col2:
        st.metric("Ready to Export", f"{len(conversations)} items")

    st.markdown("---")

    # Export format selection
    st.markdown("### Select Format")

    tab1, tab2, tab3 = st.tabs(["📄 JSON", "📊 CSV", "📕 PDF"])

    with tab1:
        st.markdown("#### JSON Export")
        st.markdown("Structured data format including full project info, documents, and conversations")

        include_docs = st.checkbox("Include document list", value=True, key="json_docs")
        include_stats = st.checkbox("Include statistics", value=True, key="json_stats")

        if st.button("Generate JSON", type="primary", key="export_json"):
            with st.spinner("Generating JSON..."):
                try:
                    exporter = ExportManager(project_id)
                    json_data = exporter.export_to_json(
                        include_documents=include_docs,
                        include_stats=include_stats
                    )

                    # Provide download button
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{project.name.replace(' ', '_')}_export.json",
                        mime="application/json"
                    )

                    st.success("✅ JSON generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with tab2:
        st.markdown("#### CSV Export")
        st.markdown("Spreadsheet format with conversation data (perfect for Excel/Google Sheets)")

        if st.button("Generate CSV", type="primary", key="export_csv"):
            with st.spinner("Generating CSV..."):
                try:
                    exporter = ExportManager(project_id)
                    csv_data = exporter.export_to_csv()

                    # Provide download button
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"{project.name.replace(' ', '_')}_conversations.csv",
                        mime="text/csv"
                    )

                    st.success("✅ CSV generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with tab3:
        st.markdown("#### PDF Export")
        st.markdown("Formatted report with all conversations (ready to print or share)")

        if st.button("Generate PDF", type="primary", key="export_pdf"):
            with st.spinner("Generating PDF..."):
                try:
                    exporter = ExportManager(project_id)
                    pdf_data = exporter.export_to_pdf()

                    # Provide download button
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"{project.name.replace(' ', '_')}_report.pdf",
                        mime="application/pdf"
                    )

                    st.success("✅ PDF generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # Preview section
    st.markdown("### Preview")

    if conversations:
        preview_count = min(3, len(conversations))
        st.markdown(f"Showing first {preview_count} of {len(conversations)} conversations:")

        for conv in conversations[:preview_count]:
            with st.expander(f"**Q:** {conv.question[:80]}..."):
                st.markdown(f"**Question:** {conv.question}")
                st.markdown(f"**Answer:** {conv.answer[:200]}...")
                st.caption(f"Date: {format_timestamp(conv.created_at)}")
    else:
        st.info("No conversations to export yet. Start chatting in the Chat tab!")


def main():
    """Main application entry point."""
    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "Projects":
        render_projects_page()
    elif page == "Knowledge Base":
        render_knowledge_base_page()
    elif page == "Chat":
        render_chat_page()
    elif page == "History":
        render_history_page()
    elif page == "Export":
        render_export_page()


if __name__ == "__main__":
    main()
