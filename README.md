# Knowledge Base

A ChromaDB-based document management system for efficient storage and retrieval of book content. This system is designed to process both structured PDFs and unstructured scanned textbooks, making their content easily searchable and retrievable.

## Features

- PDF text extraction
- OCR for scanned pages
- Text chunking and preprocessing
- Vector embeddings generation
- Semantic search capabilities
- Support for multiple document formats

## Requirements

- Python 3.8+
- Tesseract OCR engine
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlexanderDaly/knowledge-base.git
cd knowledge-base
```

2. Install Tesseract OCR:
- Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
knowledge-base/
├── src/
│   ├── processors/      # Document processing modules
│   ├── embeddings/      # Vector embedding generation
│   ├── database/        # ChromaDB operations
│   └── api/            # Query interface
├── tests/              # Unit tests
└── examples/           # Usage examples
```

## Usage

1. Process and store documents:
```python
from src.processors import pdf_processor
from src.database import chroma_manager

# Initialize database
db = chroma_manager.ChromaManager()

# Process and store a document
doc_path = "path/to/your/document.pdf"
db.add_document(doc_path)
```

2. Query the knowledge base:
```python
# Search for relevant content
results = db.query("What is the theory of relativity?")
for result in results:
    print(f"Content: {result.content}")
    print(f"Source: {result.metadata['source']}")
    print(f"Page: {result.metadata['page']}")
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
