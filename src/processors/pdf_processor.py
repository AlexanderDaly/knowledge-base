"""PDF processing module for text extraction and preprocessing."""

import PyPDF2
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF document processing and text extraction."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """Initialize the PDF processor.
        
        Args:
            chunk_size: Number of characters per text chunk
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract text from PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        try:
            path = Path(pdf_path)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                chunks = []
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text.strip():
                        logger.warning(f"No text extracted from page {page_num + 1}")
                        continue
                        
                    page_chunks = self._chunk_text(text)
                    for chunk in page_chunks:
                        chunks.append({
                            'content': chunk,
                            'metadata': {
                                'source': path.name,
                                'page': page_num + 1
                            }
                        })
                        
                logger.info(f"Processed {len(reader.pages)} pages from {path.name}")
                return chunks
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            
            # Adjust end to not split words
            if end < text_length:
                # Find the last space before chunk_size
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:  # No spaces found
                    end = start + self.chunk_size  # Force split
            
            chunks.append(text[start:end].strip())
            start = end - self.overlap

        return chunks

    def validate_pdf(self, pdf_path: str) -> bool:
        """Validate if the file is a valid PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            with open(pdf_path, 'rb') as file:
                PyPDF2.PdfReader(file)
                return True
        except Exception as e:
            logger.error(f"Invalid PDF file {pdf_path}: {str(e)}")
            return False
