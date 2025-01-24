"""ChromaDB manager for vector database operations."""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Union
import logging
import os
from pathlib import Path

from ..embeddings.embedding_manager import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages ChromaDB operations for document storage and retrieval."""
    
    def __init__(self, persist_directory: str = "./data/chroma",
                 collection_name: str = "documents",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet"
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        
        logger.info(f"Initialized ChromaDB with collection: {collection_name}")

    def add_documents(self, documents: List[Dict[str, Union[str, Dict]]]) -> None:
        """Add documents to the database.
        
        Args:
            documents: List of document dictionaries with content and metadata
        """
        try:
            # Process documents to get embeddings
            processed_docs = self.embedding_manager.process_documents(documents)
            
            # Prepare data for ChromaDB
            embeddings = [doc['embedding'] for doc in processed_docs]
            ids = [f"doc_{i}" for i in range(len(processed_docs))]
            metadatas = [doc['metadata'] for doc in processed_docs]
            documents = [doc['content'] for doc in processed_docs]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the database for similar documents.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embeddings(query_text)
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity': 1 - distance  # Convert distance to similarity
                })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_manager.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def delete_collection(self) -> None:
        """Delete the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def persist(self) -> None:
        """Persist the database to disk."""
        try:
            self.client.persist()
            logger.info("Database persisted to disk")
            
        except Exception as e:
            logger.error(f"Error persisting database: {str(e)}")
            raise
