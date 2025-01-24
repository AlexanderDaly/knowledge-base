"""Query engine for the knowledge base."""

from typing import List, Dict, Union, Optional
import logging
from pathlib import Path

from ..processors.pdf_processor import PDFProcessor
from ..processors.image_processor import ImageProcessor
from ..database.chroma_manager import ChromaManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryEngine:
    """High-level interface for the knowledge base."""
    
    def __init__(self, persist_directory: str = "./data/chroma",
                 collection_name: str = "documents",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the query engine.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
        """
        self.pdf_processor = PDFProcessor()
        self.image_processor = ImageProcessor()
        self.db = ChromaManager(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        logger.info("Initialized query engine")

    def add_document(self, file_path: Union[str, Path]) -> None:
        """Add a document to the knowledge base.
        
        Args:
            file_path: Path to the document file (PDF or image)
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Process based on file type
            if path.suffix.lower() == '.pdf':
                if not self.pdf_processor.validate_pdf(str(path)):
                    raise ValueError(f"Invalid PDF file: {file_path}")
                chunks = self.pdf_processor.extract_text(str(path))
            elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                if not self.image_processor.validate_image(str(path)):
                    raise ValueError(f"Invalid image file: {file_path}")
                chunks = self.image_processor.process_image(str(path))
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

            # Add to database
            if chunks:
                self.db.add_documents(chunks)
                self.db.persist()
                logger.info(f"Successfully added document: {path.name}")
            else:
                logger.warning(f"No content extracted from: {path.name}")

        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            raise

    def add_documents(self, directory: Union[str, Path]) -> None:
        """Add all supported documents from a directory.
        
        Args:
            directory: Path to the directory containing documents
        """
        try:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                raise NotADirectoryError(f"Not a directory: {directory}")

            # Process all supported files
            supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
            for file_path in dir_path.glob('**/*'):
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        self.add_document(file_path)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error processing directory {directory}: {str(e)}")
            raise

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the knowledge base.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            results = self.db.query(query_text, n_results)
            logger.info(f"Query completed: '{query_text}'")
            return results
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            return self.db.get_collection_stats()
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset the knowledge base by deleting all documents."""
        try:
            self.db.delete_collection()
            logger.info("Knowledge base reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting knowledge base: {str(e)}")
            raise
