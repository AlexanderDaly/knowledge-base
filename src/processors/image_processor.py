"""Image processing module for OCR and text extraction from scanned pages."""

from PIL import Image
import pytesseract
from pathlib import Path
from typing import List, Dict, Union
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing and OCR for scanned documents."""
    
    def __init__(self, tesseract_cmd: str = None, chunk_size: int = 1000, overlap: int = 200):
        """Initialize the image processor.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
            chunk_size: Number of characters per text chunk
            overlap: Number of overlapping characters between chunks
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._verify_tesseract()

    def _verify_tesseract(self):
        """Verify Tesseract OCR is properly installed."""
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error("Tesseract OCR is not properly installed or configured")
            logger.error(f"Error: {str(e)}")
            raise RuntimeError("Tesseract OCR is required but not properly configured")

    def process_image(self, image_path: Union[str, Path]) -> List[Dict[str, str]]:
        """Process an image file and extract text using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Open and preprocess image
            image = Image.open(path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            if not text.strip():
                logger.warning(f"No text extracted from image {path.name}")
                return []

            # Chunk the extracted text
            chunks = self._chunk_text(text)
            
            # Create chunks with metadata
            result = []
            for chunk in chunks:
                result.append({
                    'content': chunk,
                    'metadata': {
                        'source': path.name,
                        'type': 'scanned_page'
                    }
                })

            logger.info(f"Successfully processed image: {path.name}")
            return result

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
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

    def validate_image(self, image_path: str) -> bool:
        """Validate if the file is a valid image that can be processed.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.error(f"Invalid image file {image_path}: {str(e)}")
            return False

    @staticmethod
    def improve_image_quality(image_path: str, output_path: str = None) -> str:
        """Improve image quality for better OCR results.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the processed image (optional)
            
        Returns:
            Path to the processed image
        """
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # If no output path specified, create one
                if not output_path:
                    path = Path(image_path)
                    output_path = str(path.parent / f"processed_{path.name}")
                
                # Save the processed image
                img.save(output_path, quality=95, optimize=True)
                logger.info(f"Improved image saved to: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise
