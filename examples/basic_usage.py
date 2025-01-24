"""Example usage of the knowledge base system."""

import os
from pathlib import Path
from src.api.query_engine import QueryEngine

def main():
    # Initialize the query engine
    engine = QueryEngine(
        persist_directory="./data/chroma",
        collection_name="example_collection"
    )
    
    # Example 1: Add a single PDF document
    print("\n=== Adding a single PDF document ===")
    try:
        pdf_path = "./data/sample.pdf"  # Replace with your PDF file
        engine.add_document(pdf_path)
    except FileNotFoundError:
        print(f"Sample PDF file not found at {pdf_path}")
        print("Please add a PDF file to test this functionality")

    # Example 2: Add a scanned image
    print("\n=== Adding a scanned image ===")
    try:
        image_path = "./data/scan.png"  # Replace with your image file
        engine.add_document(image_path)
    except FileNotFoundError:
        print(f"Sample image file not found at {image_path}")
        print("Please add an image file to test this functionality")

    # Example 3: Add multiple documents from a directory
    print("\n=== Adding documents from directory ===")
    try:
        docs_dir = "./data/documents"  # Replace with your documents directory
        engine.add_documents(docs_dir)
    except NotADirectoryError:
        print(f"Documents directory not found at {docs_dir}")
        print("Please create a directory with documents to test this functionality")

    # Example 4: Query the knowledge base
    print("\n=== Querying the knowledge base ===")
    query = "What is the main topic discussed in the documents?"
    results = engine.query(query, n_results=3)
    
    print(f"\nQuery: {query}")
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {result['content'][:200]}...")  # Show first 200 chars
        print(f"Source: {result['metadata']['source']}")
        print(f"Similarity: {result['similarity']:.2f}")

    # Example 5: Get collection statistics
    print("\n=== Collection Statistics ===")
    stats = engine.get_stats()
    print(f"Document Count: {stats['document_count']}")
    print(f"Collection Name: {stats['collection_name']}")
    print(f"Embedding Dimension: {stats['embedding_dimension']}")

def setup_example_directory():
    """Create example directory structure."""
    dirs = [
        "./data",
        "./data/chroma",
        "./data/documents"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    # Setup directory structure
    setup_example_directory()
    
    # Run examples
    main()
