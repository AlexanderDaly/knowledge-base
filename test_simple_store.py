"""Test script for SimpleStore."""

from src.simple_store import SimpleStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize store
    store = SimpleStore()
    
    # Add the relativity document
    logger.info("Adding document to store...")
    store.add_document("data/documents/relativity.txt")
    
    # Get store statistics
    stats = store.get_stats()
    logger.info(f"Store statistics: {stats}")
    
    # Test queries
    test_queries = [
        "What is the principle of relativity?",
        "How is distance measured in relativity?",
        "What is the relationship between geometry and physics?",
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = store.query(query, n_results=2)
        
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i} (similarity: {result['similarity']:.3f}):")
            logger.info(f"Content: {result['content']}")
            logger.info(f"Source: {result['metadata']['source']}")
            logger.info(f"Chunk index: {result['metadata']['chunk_index']}")

if __name__ == "__main__":
    main()
