"""
PDF RAG System - Streamlit Web Interface

A web-based RAG (Retrieval-Augmented Generation) system for PDF documents
with an interactive chat interface powered by Streamlit.

Usage:
    streamlit run pdf-rag-streamlit.py
"""

import logging
import sys
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

import ollama


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the RAG pipeline.
    
    Attributes:
        model_name: Name of the custom Ollama model to create/use
        base_model: Base Ollama model to derive from
        embedding_model: Model for generating embeddings
        collection_name: Name for the Chroma collection
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between consecutive chunks
        temperature: Temperature for LLM responses
        system_prompt: System prompt for the custom model
        query_prompt_template: Template for generating query variations
        rag_prompt_template: Template for the final RAG prompt
    """
    model_name: str = "iso-iec-27001"
    base_model: str = "llama3.2:latest"
    embedding_model: str = "nomic-embed-text"
    collection_name: str = "simple-rag"
    chunk_size: int = 1200
    chunk_overlap: int = 300
    temperature: float = 0.1
    system_prompt: str = """
        you are very smart assistant who knows everything about ISO/IEC 27001. 
        You are very succinct and informative;
        You are a helpful assistant who can answer questions about ISO/IEC 27001.
        You respond `ONLY` questions about ISO/IEC 27001.
        You respond `ONLY` in English.
        If the question is not about ISO/IEC 27001, respond with: 
        "Sorry, I can only respond to questions about ISO/IEC 27001".
        Your context is limited to the content of the document you are processing: ISO/IEC 27001.
    """
    query_prompt_template: str = """
    You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from 
    a vector database. By generating multiple perspectives on the user question, your 
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by new lines.
    Original question: {question} 
    """
    rag_prompt_template: str = """Answer the question based ONLY on the following context: {context}. 
    If the question cannot be answered based on the context, respond kindly with: 
    "Sorry, I can only respond to questions about ISO/IEC 27001".
    Question: {question}
    """


def create_custom_model(config: RAGConfig) -> str:
    """Create a custom Ollama model with system prompt."""
    logger.info("Creating custom Ollama model '%s' from '%s'", config.model_name, config.base_model)
    try:
        ollama.create(
            model=config.model_name,
            from_=config.base_model,
            system=config.system_prompt,
            parameters={"temperature": config.temperature}
        )
        logger.info("Successfully created model '%s'", config.model_name)
        return config.model_name
    except Exception as e:
        logger.error("Failed to create custom model: %s", e)
        raise RuntimeError(f"Failed to create custom model '{config.model_name}': {e}") from e


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
) -> Chroma:
    """Create a Chroma vector store from document chunks."""
    logger.info("Pulling embedding model: %s", embedding_model)
    try:
        ollama.pull(embedding_model)
        logger.info("Embedding model pulled successfully")
    except Exception as e:
        logger.error("Failed to pull embedding model: %s", e)
        raise RuntimeError(f"Failed to pull embedding model '{embedding_model}': {e}") from e
    
    logger.info("Creating vector store with collection '%s' (%d chunks)", collection_name, len(chunks))
    try:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=embedding_model),
            collection_name=collection_name
        )
        logger.info("Successfully created vector store")
        return vector_db
    except Exception as e:
        logger.error("Failed to create vector store: %s", e)
        raise RuntimeError(f"Failed to create vector store: {e}") from e


def create_retriever(
    vector_db: Chroma,
    llm: BaseChatModel,
    query_template: str
) -> BaseRetriever:
    """Create a multi-query retriever from the vector store."""
    logger.info("Creating multi-query retriever")
    try:
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=query_template
        )
        
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            llm,
            prompt=query_prompt
        )
        logger.info("Successfully created retriever")
        return retriever
    except Exception as e:
        logger.error("Failed to create retriever: %s", e)
        raise RuntimeError(f"Failed to create retriever: {e}") from e


def create_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    rag_template: str
) -> Runnable:
    """Create the RAG processing chain."""
    logger.info("Creating RAG chain")
    try:
        prompt = ChatPromptTemplate.from_template(rag_template)
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Successfully created RAG chain")
        return chain
    except Exception as e:
        logger.error("Failed to create RAG chain: %s", e)
        raise RuntimeError(f"Failed to create RAG chain: {e}") from e


def initialize_rag_pipeline(config: RAGConfig, doc_path: str) -> Runnable:
    """Initialize the RAG pipeline with a specific document."""
    logger.info("Initializing RAG pipeline for: %s", doc_path)
    
    model_name = create_custom_model(config)
    documents = load_document(doc_path)
    chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
    vector_db = create_vector_store(
        chunks,
        config.embedding_model,
        config.collection_name
    )
    
    llm = ChatOllama(model=model_name)
    retriever = create_retriever(vector_db, llm, config.query_prompt_template)
    chain = create_rag_chain(retriever, llm, config.rag_prompt_template)
    
    logger.info("RAG pipeline initialized successfully")
    return chain


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PDF RAG Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 PDF RAG Chat System")
    st.markdown("Ask questions about your PDF documents using AI-powered retrieval.")
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to index and chat with"
        )
        
        st.divider()
        
        st.header("⚙️ Settings")
        chunk_size = st.slider("Chunk Size", 200, 2000, 1200, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 300, 50)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chain" not in st.session_state:
        st.session_state.chain = None
    
    if "doc_processed" not in st.session_state:
        st.session_state.doc_processed = False
    
    # Process uploaded document
    if uploaded_file is not None and not st.session_state.doc_processed:
        with st.spinner("Processing document... This may take a moment."):
            try:
                # Save uploaded file temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Create config with user settings
                config = RAGConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    temperature=temperature
                )
                
                # Initialize pipeline
                st.session_state.chain = initialize_rag_pipeline(config, temp_path)
                st.session_state.doc_processed = True
                
                st.success(f"✅ Document '{uploaded_file.name}' indexed successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to process document: {e}")
                logger.error("Document processing failed: %s", e)
    
    # Display chat interface
    if st.session_state.doc_processed and st.session_state.chain is not None:
        # Show chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chain.invoke(input=prompt)
                        st.markdown(response)
                        
                        # Add assistant response to history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        logger.error("Chat response generation failed: %s", e)
    
    else:
        # Show instructions when no document is loaded
        st.info("👆 Please upload a PDF document in the sidebar to start chatting.")
        
        st.markdown("""
        ### How to use:
        1. **Upload a PDF** using the sidebar file uploader
        2. **Adjust settings** (optional) - chunk size, overlap, temperature
        3. **Ask questions** in the chat box
        4. The AI will retrieve relevant content from your document and answer
        
        ### Features:
        - 💬 Interactive chat interface
        - 🔍 AI-powered document retrieval
        - ⚙️ Adjustable processing parameters
        - 📝 Persistent chat history per session
        """)


if __name__ == "__main__":
    main()
