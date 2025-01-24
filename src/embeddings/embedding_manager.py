"""Embedding manager for generating and managing vector embeddings."""

from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages the generation and handling of text embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for one or more texts.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def process_documents(self, documents: List[Dict[str, str]]) -> List[Dict]:
        """Process a list of documents and add embeddings.
        
        Args:
            documents: List of document dictionaries with 'content' key
            
        Returns:
            List of documents with added embeddings
        """
        try:
            # Extract text content
            texts = [doc['content'] for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to documents
            processed_docs = []
            for doc, embedding in zip(documents, embeddings):
                doc_with_embedding = doc.copy()
                doc_with_embedding['embedding'] = embedding
                processed_docs.append(doc_with_embedding)
                
            logger.info(f"Generated embeddings for {len(processed_docs)} documents")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize the embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Compute cosine similarity
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise

    def batch_compute_similarity(self, query_embedding: np.ndarray, 
                               document_embeddings: List[np.ndarray]) -> List[float]:
        """Compute similarities between a query and multiple documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
        """
        try:
            similarities = []
            for doc_embedding in document_embeddings:
                similarity = self.compute_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing batch similarities: {str(e)}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()
