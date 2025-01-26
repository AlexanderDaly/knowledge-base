from src.simple_store import SimpleStore

def main():
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
        "How does the chunking work?"
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
