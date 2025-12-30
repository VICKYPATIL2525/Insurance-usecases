# Statement 6: Document Classifier

## Description
Classifies insurance documents using embedding similarity against reference documents.

## Files
- `statement_6_dbmaker.py` - Creates vector database from reference PDFs
- `statement_6_main.py` - Classifies test documents

## Data Structure
- `data/vector_db/` - Reference PDF documents for classification
- `data/st_test_1/` - Test PDFs to classify
- `data/testing/` - Additional test documents

## Setup
1. First, create the vector database from reference documents:
   ```bash
   python statement_6_dbmaker.py
   ```

2. Then classify test documents:
   ```bash
   python statement_6_main.py
   ```

## How It Works
1. Reference PDFs are converted to embeddings and stored in ChromaDB
2. Test PDFs are compared against reference documents
3. Classification based on highest similarity score

## Database
Uses shared vector database at `../shared/chroma_store/`
