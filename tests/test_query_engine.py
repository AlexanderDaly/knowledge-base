"""Tests for the knowledge base query engine."""

import pytest
from pathlib import Path
import tempfile
import shutil
import os

from src.api.query_engine import QueryEngine
from src.processors.pdf_processor import PDFProcessor
from src.processors.image_processor import ImageProcessor

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def query_engine(temp_dir):
    """Create a query engine instance for testing."""
    persist_dir = os.path.join(temp_dir, "chroma")
    engine = QueryEngine(
        persist_directory=persist_dir,
        collection_name="test_collection"
    )
    return engine

def test_query_engine_initialization(query_engine):
    """Test that the query engine initializes correctly."""
    assert query_engine.pdf_processor is not None
    assert query_engine.image_processor is not None
    assert query_engine.db is not None

def test_get_stats_empty_collection(query_engine):
    """Test getting stats for an empty collection."""
    stats = query_engine.get_stats()
    assert stats['document_count'] == 0
    assert stats['collection_name'] == "test_collection"
    assert stats['embedding_dimension'] > 0

def test_invalid_file_handling(query_engine, temp_dir):
    """Test handling of invalid files."""
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        query_engine.add_document("nonexistent.pdf")

    # Test unsupported file type
    invalid_file = os.path.join(temp_dir, "test.txt")
    with open(invalid_file, 'w') as f:
        f.write("test content")
    
    with pytest.raises(ValueError) as exc_info:
        query_engine.add_document(invalid_file)
    assert "Unsupported file type" in str(exc_info.value)

def test_query_empty_collection(query_engine):
    """Test querying an empty collection."""
    results = query_engine.query("test query")
    assert isinstance(results, list)
    assert len(results) == 0

def test_reset_collection(query_engine):
    """Test resetting the collection."""
    # Get initial stats
    initial_stats = query_engine.get_stats()
    
    # Reset collection
    query_engine.reset()
    
    # Get stats after reset
    final_stats = query_engine.get_stats()
    
    assert final_stats['document_count'] == 0
    assert final_stats['collection_name'] == initial_stats['collection_name']

@pytest.mark.integration
def test_pdf_processor_initialization():
    """Test PDF processor initialization."""
    processor = PDFProcessor()
    assert processor.chunk_size == 1000
    assert processor.overlap == 200

@pytest.mark.integration
def test_image_processor_initialization():
    """Test image processor initialization."""
    processor = ImageProcessor()
    assert processor.chunk_size == 1000
    assert processor.overlap == 200

def test_invalid_directory_handling(query_engine):
    """Test handling of invalid directories."""
    with pytest.raises(NotADirectoryError):
        query_engine.add_documents("nonexistent_directory")

# Add these test cases when you have sample files to test with
"""
@pytest.mark.integration
def test_pdf_processing(query_engine, temp_dir):
    # Create a sample PDF and test processing
    pass

@pytest.mark.integration
def test_image_processing(query_engine, temp_dir):
    # Create a sample image and test processing
    pass

@pytest.mark.integration
def test_full_workflow(query_engine, temp_dir):
    # Test the complete workflow from adding documents to querying
    pass
"""
