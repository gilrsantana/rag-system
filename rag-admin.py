"""
RAG Admin - Document Processing Interface

Admin interface for uploading, processing, and indexing PDF documents
for the RAG system. Creates persistent vector stores that users can query.

Usage:
    streamlit run rag-admin.py
"""

import logging
import sys
import os
import hashlib
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Setup logging first (before any logger usage)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

import ollama

# Ollama configuration for Docker
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
if OLLAMA_BASE_URL != "http://localhost:11434":
    ollama.Client(host=OLLAMA_BASE_URL)
    logger.info("Using Ollama at: %s", OLLAMA_BASE_URL)


VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_stores")
METADATA_FILE = os.getenv("METADATA_FILE", "metadata.json")


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for document processing.
    
    Attributes:
        embedding_model: Model for generating embeddings
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between consecutive chunks
    """
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 1200
    chunk_overlap: int = 300


def ensure_vector_store_dir():
    """Ensure the vector store directory exists."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


def generate_collection_name(filename: str) -> str:
    """Generate a unique collection name from filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    safe_name = Path(filename).stem.replace(" ", "_").lower()
    return f"{safe_name}_{name_hash}_{timestamp}"


def save_metadata(collection_name: str, metadata: dict) -> None:
    """Save document metadata to JSON file."""
    import json
    store_path = os.path.join(VECTOR_STORE_DIR, collection_name)
    metadata_path = os.path.join(store_path, METADATA_FILE)
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Saved metadata for: %s", collection_name)
    except Exception as e:
        logger.error("Failed to save metadata: %s", e)


def load_metadata(collection_name: str) -> Optional[dict]:
    """Load document metadata from JSON file."""
    import json
    metadata_path = os.path.join(VECTOR_STORE_DIR, collection_name, METADATA_FILE)
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load metadata: %s", e)
    return None


def get_ollama_client():
    """Get Ollama client with configured host."""
    if OLLAMA_BASE_URL != "http://localhost:11434":
        return ollama.Client(host=OLLAMA_BASE_URL)
    return ollama


def generate_document_summary(documents: List[Document], model: str = "llama3.2:latest") -> dict:
    """Auto-generate title and description from document content."""
    client = get_ollama_client()
    logger.info("Generating document summary using LLM at %s", OLLAMA_BASE_URL)
    
    # Sample content from first few pages
    sample_text = "\n\n".join([doc.page_content[:1500] for doc in documents[:3]])
    
    prompt = f"""Based on the following document excerpt, generate:
1. A concise title (max 5 words)
2. A brief description (max 2 sentences) explaining what this document is about

Document excerpt:
{sample_text[:4000]}

Respond in this exact format:
Title: <your title>
Description: <your description>
"""
    
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response['message']['content']
        
        # Parse response
        title = "Untitled Document"
        description = "No description available"
        
        for line in content.split('\n'):
            if line.lower().startswith('title:'):
                title = line.split(':', 1)[1].strip()
            elif line.lower().startswith('description:'):
                description = line.split(':', 1)[1].strip()
        
        logger.info("Generated title: %s", title)
        return {"title": title, "description": description}
    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        return {
            "title": "Untitled Document",
            "description": "Document uploaded without automatic description generation."
        }


def get_directory_size(path: str) -> float:
    """Get total size of directory in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)


def list_indexed_documents() -> List[dict]:
    """List all indexed documents in the vector store directory."""
    ensure_vector_store_dir()
    documents = []

    for item in os.listdir(VECTOR_STORE_DIR):
        item_path = os.path.join(VECTOR_STORE_DIR, item)
        if os.path.isdir(item_path):
            stat = os.stat(item_path)
            metadata = load_metadata(item)
            size_mb = get_directory_size(item_path)

            doc_info = {
                "name": item,
                "path": item_path,
                "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_mb": size_mb,
                "title": metadata.get("title", "Untitled") if metadata else "Untitled",
                "description": metadata.get("description", "No description") if metadata else "No description"
            }
            documents.append(doc_info)

    return sorted(documents, key=lambda x: x["created"], reverse=True)


def delete_document(collection_name: str) -> bool:
    """Delete an indexed document."""
    try:
        doc_path = os.path.join(VECTOR_STORE_DIR, collection_name)
        if os.path.exists(doc_path):
            import shutil
            shutil.rmtree(doc_path)
            logger.info("Deleted document collection: %s", collection_name)
            return True
    except Exception as e:
        logger.error("Failed to delete document %s: %s", collection_name, e)
    return False


def load_document(doc_path: str, languages: List[str] = None) -> List[Document]:
    """Load a PDF document from the specified path."""
    if not doc_path:
        raise ValueError("Please provide a valid document path.")
    
    if languages is None:
        languages = ["eng"]
    
    logger.info("Loading document from: %s", doc_path)
    try:
        loader = UnstructuredPDFLoader(doc_path, languages=languages)
        documents = loader.load()
        logger.info("Successfully loaded %d document(s)", len(documents))
        return documents
    except FileNotFoundError as e:
        logger.error("Document not found: %s", doc_path)
        raise RuntimeError(f"Document not found: {doc_path}") from e
    except Exception as e:
        logger.error("Failed to load document: %s", e)
        raise RuntimeError(f"Failed to load document from '{doc_path}': {e}") from e


def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """Split documents into smaller chunks."""
    logger.info("Splitting %d document(s) into chunks (size=%d, overlap=%d)", 
                len(documents), chunk_size, chunk_overlap)
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logger.info("Successfully split into %d chunks", len(chunks))
        return chunks
    except Exception as e:
        logger.error("Failed to split documents: %s", e)
        raise RuntimeError(f"Failed to split documents: {e}") from e


def create_vector_store(
    chunks: List[Document],
    embedding_model: str,
    collection_name: str
) -> str:
    """Create a Chroma vector store from document chunks.
    
    Returns:
        Path to the created vector store
    """
    client = get_ollama_client()
    logger.info("Pulling embedding model: %s", embedding_model)
    try:
        client.pull(embedding_model)
        logger.info("Embedding model pulled successfully")
    except Exception as e:
        logger.error("Failed to pull embedding model: %s", e)
        raise RuntimeError(f"Failed to pull embedding model '{embedding_model}': {e}") from e
    
    ensure_vector_store_dir()
    persist_path = os.path.join(VECTOR_STORE_DIR, collection_name)
    
    logger.info("Creating vector store at: %s (%d chunks)", persist_path, len(chunks))
    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_BASE_URL),
            collection_name=collection_name,
            persist_directory=persist_path
        )
        logger.info("Successfully created and persisted vector store")
        return persist_path
    except Exception as e:
        logger.error("Failed to create vector store: %s", e)
        raise RuntimeError(f"Failed to create vector store: {e}") from e


def process_document(
    file_path: str,
    filename: str,
    config: ProcessingConfig,
    custom_title: str = None,
    custom_description: str = None,
    auto_generate: bool = True
) -> dict:
    """Process a document and create a vector store.
    
    Returns:
        Dict with processing results including collection name
    """
    collection_name = generate_collection_name(filename)
    
    documents = load_document(file_path)
    chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    store_path = create_vector_store(
        chunks,
        config.embedding_model,
        collection_name
    )
    
    # Generate or use provided metadata
    if custom_title and custom_description:
        metadata = {
            "title": custom_title,
            "description": custom_description,
            "filename": filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }
    elif auto_generate:
        metadata = generate_document_summary(documents, config.embedding_model)
        metadata.update({
            "filename": filename,
            "pages": len(documents),
            "chunks": len(chunks)
        })
    else:
        metadata = {
            "title": custom_title or Path(filename).stem,
            "description": custom_description or "No description provided",
            "filename": filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }
    
    save_metadata(collection_name, metadata)
    
    return {
        "collection_name": collection_name,
        "store_path": store_path,
        "chunks": len(chunks),
        "pages": len(documents),
        "metadata": metadata
    }


def main():
    """Main Streamlit admin application."""
    st.set_page_config(
        page_title="RAG Admin - Document Processing",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 RAG Admin - Document Processing")
    st.markdown("Upload and index PDF documents for the RAG system.")
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Processing Settings")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=1200,
            step=100,
            help="Number of characters per chunk"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=300,
            step=50,
            help="Number of overlapping characters between chunks"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["nomic-embed-text", "llama3.2:latest"],
            index=0,
            help="Model used to generate embeddings"
        )
        
        st.divider()
        st.info(f"Vector stores are saved to:\n`{os.path.abspath(VECTOR_STORE_DIR)}`")
    
    # Main content tabs
    tab_upload, tab_manage = st.tabs(["📤 Upload Document", "📚 Manage Documents"])
    
    # Upload Tab
    with tab_upload:
        st.subheader("Upload New Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file to index",
            type="pdf",
            help="Upload a PDF document to process and index"
        )
        
        if uploaded_file is not None:
            st.info(f"**Selected file:** {uploaded_file.name}")
            st.text(f"File size: {uploaded_file.size / 1024:.1f} KB")
            
            st.divider()
            
            # Document Information Section
            st.subheader("📝 Document Information")
            
            auto_generate = st.toggle(
                "🤖 Auto-generate title and description",
                value=False,
                help="Use AI to automatically generate title and description from document content"
            )
            
            custom_title = None
            custom_description = None
            
            if not auto_generate:
                custom_title = st.text_input(
                    "Document Title",
                    placeholder="e.g., ISO 27001 Security Standard",
                    help="Enter a descriptive title for this document"
                )
                custom_description = st.text_area(
                    "Document Description",
                    placeholder="Describe what this document is about...",
                    help="Enter a brief description to help users understand this document's content",
                    max_chars=500
                )
            
            st.divider()
            
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing document... This may take a few minutes."):
                    try:
                        # Save temporarily
                        temp_path = f"/tmp/{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Process
                        config = ProcessingConfig(
                            embedding_model=embedding_model,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        result = process_document(
                            temp_path, 
                            uploaded_file.name, 
                            config,
                            custom_title=custom_title,
                            custom_description=custom_description,
                            auto_generate=auto_generate
                        )
                        
                        # Success message
                        st.success("✅ Document processed successfully!")
                        
                        # Display metadata
                        st.subheader("📄 Document Metadata")
                        st.markdown(f"**Title:** {result['metadata']['title']}")
                        st.markdown(f"**Description:** {result['metadata']['description']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pages", result["pages"])
                        with col2:
                            st.metric("Chunks", result["chunks"])
                        with col3:
                            st.metric("Collection ID", result["collection_name"][:15] + "...")
                        
                        st.code(result["collection_name"], language="text")
                        st.info("💡 Users will see the title and description when selecting documents")
                        
                        # Cleanup
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Failed to process document: {e}")
                        logger.error("Document processing failed: %s", e)
    
    # Manage Tab
    with tab_manage:
        st.subheader("Indexed Documents")
        
        documents = list_indexed_documents()
        
        if not documents:
            st.info("No documents indexed yet. Upload a document to get started.")
        else:
            st.write(f"**{len(documents)}** document(s) indexed")
            
            for doc in documents:
                with st.expander(f"📄 {doc['title']}"):
                    st.markdown(f"**Description:** {doc['description']}")
                    st.markdown(f"**Collection:** `{doc['name']}`")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.text(f"Created: {doc['created']}")
                    with col2:
                        st.text(f"Size: {doc['size_mb']} MB")
                    with col3:
                        if st.button("🗑️ Delete", key=f"del_{doc['name']}"):
                            if delete_document(doc['name']):
                                st.success("Deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete")


if __name__ == "__main__":
    main()
