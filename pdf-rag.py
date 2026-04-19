"""
PDF RAG System

A modular implementation of a Retrieval-Augmented Generation system for PDF documents.

Components:
    1. PDF Ingestion - Load PDF documents
    2. Text Extraction & Chunking - Split documents into manageable chunks
    3. Embedding Generation - Create embeddings using Ollama
    4. Vector Storage - Store embeddings in Chroma DB
    5. Similarity Search - Retrieve relevant documents
    6. RAG Pipeline - Answer questions using retrieved context

Usage:
    python pdf-rag.py
"""

import logging
import sys
from typing import List
from dataclasses import dataclass

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
        doc_path: Path to the PDF document
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
    doc_path: str = "./data/ISO_IEC-270012022-ed.3.pdf"
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
    """Create a custom Ollama model with system prompt.
    
    Args:
        config: RAG configuration containing model parameters
        
    Returns:
        Name of the created model
        
    Raises:
        RuntimeError: If model creation fails
    """
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
    """Load a PDF document from the specified path.
    
    Args:
        doc_path: Path to the PDF file
        languages: List of languages for document processing
        
    Returns:
        List of loaded documents
        
    Raises:
        ValueError: If doc_path is empty or None
        RuntimeError: If document loading fails
    """
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
    """Split documents into smaller chunks.
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of document chunks
        
    Raises:
        RuntimeError: If document splitting fails
    """
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
    """Create a Chroma vector store from document chunks.
    
    Args:
        chunks: List of document chunks to embed
        embedding_model: Name of the Ollama embedding model
        collection_name: Name for the Chroma collection
        
    Returns:
        Configured Chroma vector store
        
    Raises:
        RuntimeError: If vector store creation fails
    """
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
    """Create a multi-query retriever from the vector store.
    
    Args:
        vector_db: Chroma vector store to query
        llm: Language model for query expansion
        query_template: Template for generating query variations
        
    Returns:
        Configured MultiQueryRetriever
        
    Raises:
        RuntimeError: If retriever creation fails
    """
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
    """Create the RAG processing chain.
    
    Args:
        retriever: Document retriever for fetching context
        llm: Language model for generating responses
        rag_template: Template for the final RAG prompt
        
    Returns:
        Configured RAG chain
        
    Raises:
        RuntimeError: If chain creation fails
    """
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


def answer_question(chain: Runnable, question: str) -> str:
    """Invoke the RAG chain with a question.
    
    Args:
        chain: Configured RAG chain
        question: User's question
        
    Returns:
        Generated answer
        
    Raises:
        RuntimeError: If query execution fails
    """
    logger.info("Processing question: %s", question)
    try:
        response = chain.invoke(input=question)
        logger.info("Successfully generated response")
        return response
    except Exception as e:
        logger.error("Failed to process question: %s", e)
        raise RuntimeError(f"Failed to process question: {e}") from e


def initialize_rag_pipeline(config: RAGConfig) -> Runnable:
    """Initialize the RAG pipeline components.
    
    This function performs all one-time setup operations including:
    - Creating the custom Ollama model
    - Loading and chunking the document
    - Building the vector store
    - Setting up the retriever and RAG chain
    
    Args:
        config: RAG configuration
        
    Returns:
        Configured RAG chain ready for querying
        
    Raises:
        RuntimeError: If any initialization step fails
    """
    logger.info("Initializing RAG pipeline...")
    
    model_name = create_custom_model(config)
    documents = load_document(config.doc_path)
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


def run_chat_session(chain: Runnable) -> None:
    """Run an interactive CLI chat session.
    
    Continuously prompts the user for questions until they type
    'quit', 'exit', or 'q'.
    
    Args:
        chain: Configured RAG chain for answering questions
    """
    print("\n" + "=" * 60)
    print("RAG Chat Session Started")
    print("Document indexed and ready for questions.")
    print("Type 'quit', 'exit', or 'q' to end the session.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("quit", "exit", "q"):
                print("\nEnding chat session. Goodbye!")
                break
            
            response = answer_question(chain, question)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Ending chat session. Goodbye!")
            break
        except Exception as e:
            logger.error("Error during chat: %s", e)
            print(f"\nError: Unable to process your question. Please try again.\n")


def main():
    """Main entry point for the RAG pipeline."""
    logger.info("Starting RAG application")
    
    config = RAGConfig()
    
    try:
        chain = initialize_rag_pipeline(config)
        run_chat_session(chain)
        logger.info("RAG application completed successfully")
    except Exception as e:
        logger.error("RAG pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

