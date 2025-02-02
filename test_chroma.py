import sys
from src.simple_store import SimpleStore

def main():
    # Set console to UTF-8 mode
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    # Initialize store
    store = SimpleStore()
    
    # Add test document
    store.add_document('test.txt')
    
    # Print store stats
    stats = store.get_stats()
    print(f"\nStore stats: {stats}\n")
    
    # Test queries
    queries = [
        "What does this document test?",
        "What capabilities does ChromaDB provide?",
        "What is the 75th Pangram?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = store.query(query)
        for result in results:
            print(f"\nContent: {result['content']}")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Source: {result['metadata']['source']}")

if __name__ == '__main__':
    main()
