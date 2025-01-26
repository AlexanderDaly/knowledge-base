"""Vector store implementation using ChromaDB and sentence-transformers."""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleStore:
    """A vector store for text documents using ChromaDB."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the store.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.Client(Settings(
            persist_directory="./data/chroma_db",
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=lambda texts: self.model.encode(texts).tolist()
        )
        
        logger.info(f"Initialized SimpleStore with model: {model_name}")

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            
            # Adjust end to not split words
            if end < text_length:
                # Find the last space before chunk_size
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:  # No spaces found
                    end = start + chunk_size  # Force split
            
            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks

    def add_document(self, file_path: str) -> None:
        """Add a document to the store.
        
        Args:
            file_path: Path to the text file
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Split into chunks
            chunks = self._chunk_text(text)
            
            # Generate IDs and metadata for chunks
            ids = [f"{path.name}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    'source': path.name,
                    'chunk_index': i
                }
                for i in range(len(chunks))
            ]
            
            # Add to ChromaDB collection
            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Added document {path.name} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            raise

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the store for similar text chunks.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        try:
            # Query ChromaDB collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results to match original API
            formatted_results = []
            if results['documents']:
                for doc, meta, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    # Convert distance to similarity score (ChromaDB returns L2 distance)
                    # Normalize to 0-1 range where 1 is most similar
                    similarity = 1 / (1 + distance)
                    
                    formatted_results.append({
                        'content': doc,
                        'metadata': meta,
                        'similarity': float(similarity)
                    })
                    
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying store: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """Get statistics about the store.
        
        Returns:
            Dictionary containing statistics
        """
        count = self.collection.count()
        return {
            'document_count': count,
            'embedding_dimension': self.model.get_sentence_embedding_dimension()
        }
