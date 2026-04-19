"""
RAG User - Document Query Interface

User interface for querying indexed PDF documents.
Connects to vector stores created by the admin interface.

Usage:
    streamlit run rag-user.py
"""

import logging
import sys
import os
from typing import List, Optional
from dataclasses import dataclass

import streamlit as st
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

import ollama

# Setup logging first (before any logger usage)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ollama configuration for Docker
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
if OLLAMA_BASE_URL != "http://localhost:11434":
    ollama.Client(host=OLLAMA_BASE_URL)
    logger.info("Using Ollama at: %s", OLLAMA_BASE_URL)


def get_ollama_client():
    """Get Ollama client with configured host."""
    if OLLAMA_BASE_URL != "http://localhost:11434":
        return ollama.Client(host=OLLAMA_BASE_URL)
    return ollama


VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_stores")
METADATA_FILE = os.getenv("METADATA_FILE", "metadata.json")


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for querying documents.
    
    Attributes:
        model_name: Name of the custom Ollama model to use
        base_model: Base Ollama model to derive from
        embedding_model: Model for embeddings (must match admin)
        temperature: Temperature for LLM responses
        system_prompt: System prompt for the assistant
        query_prompt_template: Template for generating query variations
        rag_prompt_template: Template for the final RAG prompt
    """
    model_name: str = "rag-assistant"
    base_model: str = "llama3.2:latest"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.1
    system_prompt: str = """
        You are a helpful assistant that answers questions based on provided documents.
        You are succinct and informative in your responses.
        CRITICAL: You must ONLY use information from the provided context.
        You have NO knowledge of anything outside the provided documents.
        If the context does not contain the answer, you MUST respond with:
        "I apologize, but I cannot find the answer to that question in the provided documents."
        Do NOT use any general knowledge. Do NOT make up answers. Stay strictly within the provided context.
    """
    query_prompt_template: str = """
    You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from 
    a vector database. By generating multiple perspectives on the user question, your 
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by new lines.
    Original question: {question} 
    """
    rag_prompt_template: str = """You are a retrieval-based question answering system.
    You have access to ONLY the following context - you know nothing else:

    Context:
    {context}

    INSTRUCTIONS:
    - Use ONLY the information in the context above
    - If the context contains the answer, provide it concisely
    - If the context does NOT contain the answer, you MUST respond EXACTLY with:
      "I apologize, but I cannot find the answer to that question in the provided documents."
    - Do NOT use any outside knowledge
    - Do NOT make up or infer information not in the context
    - Do NOT answer general knowledge questions

    Question: {question}

    Answer (use only the context provided):"""


def ensure_vector_store_dir():
    """Ensure the vector store directory exists."""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


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


def list_available_documents() -> List[dict]:
    """List all available documents with their metadata."""
    ensure_vector_store_dir()
    documents = []
    
    if os.path.exists(VECTOR_STORE_DIR):
        for item in os.listdir(VECTOR_STORE_DIR):
            item_path = os.path.join(VECTOR_STORE_DIR, item)
            if os.path.isdir(item_path):
                metadata = load_metadata(item)
                if metadata:
                    documents.append({
                        "collection_name": item,
                        "title": metadata.get("title", "Untitled"),
                        "description": metadata.get("description", "No description"),
                        "pages": metadata.get("pages", 0),
                        "chunks": metadata.get("chunks", 0)
                    })
                else:
                    documents.append({
                        "collection_name": item,
                        "title": item,
                        "description": "No description available",
                        "pages": 0,
                        "chunks": 0
                    })
    
    return sorted(documents, key=lambda x: x["title"])


def create_custom_model(config: QueryConfig) -> str:
    """Create a custom Ollama model with system prompt."""
    client = get_ollama_client()
    logger.info("Creating custom Ollama model '%s' from '%s' at %s", 
                config.model_name, config.base_model, OLLAMA_BASE_URL)
    try:
        client.create(
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


def load_vector_store(collection_name: str, embedding_model: str) -> Chroma:
    """Load an existing Chroma vector store."""
    store_path = os.path.join(VECTOR_STORE_DIR, collection_name)
    
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Vector store not found: {store_path}")
    
    logger.info("Loading vector store from: %s", store_path)
    try:
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_BASE_URL),
            persist_directory=store_path
        )
        logger.info("Successfully loaded vector store")
        return vector_db
    except Exception as e:
        logger.error("Failed to load vector store: %s", e)
        raise RuntimeError(f"Failed to load vector store: {e}") from e


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


def initialize_query_engine(collection_name: str, config: QueryConfig) -> Runnable:
    """Initialize the query engine for a specific collection."""
    logger.info("Initializing query engine for collection: %s", collection_name)
    
    # Create/pull model
    model_name = create_custom_model(config)
    
    # Load existing vector store
    vector_db = load_vector_store(collection_name, config.embedding_model)
    
    # Setup LLM and retriever
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)
    retriever = create_retriever(vector_db, llm, config.query_prompt_template)
    chain = create_rag_chain(retriever, llm, config.rag_prompt_template)
    
    logger.info("Query engine initialized successfully")
    return chain


def answer_question(chain: Runnable, question: str) -> str:
    """Invoke the RAG chain with a question."""
    logger.info("Processing question: %s", question)
    try:
        response = chain.invoke(input=question)
        logger.info("Successfully generated response")
        return response
    except Exception as e:
        logger.error("Failed to process question: %s", e)
        raise RuntimeError(f"Failed to process question: {e}") from e


def main():
    """Main Streamlit user application."""
    st.set_page_config(
        page_title="RAG User - Document Query",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💬 RAG User - Document Query")
    st.markdown("Ask questions about indexed documents.")
    
    # Sidebar - Document Selection
    with st.sidebar:
        st.header("📚 Select Document")
        
        available_documents = list_available_documents()
        
        if not available_documents:
            st.warning("⚠️ No documents available. Please ask an administrator to index documents first.")
            selected_document = None
        else:
            # Create dropdown options with titles
            doc_options = {doc["collection_name"]: f"📄 {doc['title']}" for doc in available_documents}
            
            selected_collection = st.selectbox(
                "Choose a document",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
                help="Select from available indexed documents"
            )
            
            selected_document = next(
                (doc for doc in available_documents if doc["collection_name"] == selected_collection),
                None
            )
            
            if selected_document:
                st.divider()
                st.subheader("📄 Document Information")
                st.markdown(f"**{selected_document['title']}**")
                st.caption(selected_document['description'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pages", selected_document['pages'])
                with col2:
                    st.metric("Chunks", selected_document['chunks'])
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area
    if not available_documents:
        st.info("👈 No documents available. Please contact an administrator to upload and index documents.")
        
        st.markdown("""
        ### How It Works:
        1. An **administrator** runs the admin interface to upload and index PDF documents
        2. Once indexed, documents appear in this interface
        3. You select a document and start asking questions
        
        ### Administrator Interface:
        ```bash
        streamlit run rag-admin.py
        ```
        """)
        return
    
    if selected_document:
        selected_collection = selected_document["collection_name"]
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "current_collection" not in st.session_state:
            st.session_state.current_collection = None
        
        if "chain" not in st.session_state:
            st.session_state.chain = None
        
        # Reset if collection changed
        if st.session_state.current_collection != selected_collection:
            st.session_state.messages = []
            st.session_state.current_collection = selected_collection
            st.session_state.chain = None
        
        # Display document info header
        st.subheader(f"📄 {selected_document['title']}")
        st.caption(selected_document['description'])
        st.divider()
        
        # Initialize chain if needed
        if st.session_state.chain is None:
            with st.spinner("Loading document..."):
                try:
                    config = QueryConfig()
                    st.session_state.chain = initialize_query_engine(selected_collection, config)
                except Exception as e:
                    st.error(f"❌ Failed to load document: {e}")
                    logger.error("Failed to initialize query engine: %s", e)
                    return
        
        # Display chat interface
        st.markdown("### 💬 Ask Questions")
        
        # Show chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about this document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = answer_question(st.session_state.chain, prompt)
                        st.markdown(response)
                        
                        # Add to history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        error_msg = f"I apologize, but I encountered an error: {str(e)}"
                        st.error(error_msg)
                        logger.error("Chat response generation failed: %s", e)


if __name__ == "__main__":
    main()
