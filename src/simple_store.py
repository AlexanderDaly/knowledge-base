"""Simple vector store implementation using sentence-transformers and numpy."""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleStore:
    """A simple vector store for text documents."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the store.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []  # List of document chunks
        self.embeddings = None  # Document embeddings
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
            
            # Add metadata
            doc_chunks = [
                {
                    'content': chunk,
                    'metadata': {
                        'source': path.name,
                        'chunk_index': i
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Generate embeddings
            new_embeddings = self.model.encode([chunk['content'] for chunk in doc_chunks])
            
            # Add to store
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
                
            self.documents.extend(doc_chunks)
            
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
            if not self.documents:
                return []
                
            # Generate query embedding
            query_embedding = self.model.encode([query_text])[0]
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top results
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'content': self.documents[idx]['content'],
                    'metadata': self.documents[idx]['metadata'],
                    'similarity': float(similarities[idx])
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error querying store: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """Get statistics about the store.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            'document_count': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0
        }
