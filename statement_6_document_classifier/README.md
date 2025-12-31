# Statement 6: Document Classifier

## Description
Classifies insurance documents using embedding similarity against reference documents. Uses RAG-based approach to determine document types.

## Files
- `chroma_dbmaker.py` - Creates vector database from reference PDFs
- `main.py` - Classifies test documents (CLI version)
- `app.py` - Flask web UI for document classification
- `templates/document_classifier.html` - Web interface HTML
- `uploads/` - Folder for user-uploaded documents (web UI)
- `data/vector_db/` - Reference documents (3 PDFs)
- `data/st_test_1/` - Test set 1 (4 PDFs)
- `data/testing/` - Test set 2 (15 PDFs)

## Data Structure

### Reference Documents (vector_db/)
These are the "ground truth" documents used for classification:
- `claim_form_vector.pdf` - Standard insurance claim form
- `inspection_report_vector.pdf` - Property inspection report
- `invoice_vector.pdf` - Service invoice/bill

### Test Documents
**Test Set 1** (`st_test_1/` - 4 PDFs):
- Various claim forms and invoices

**Test Set 2** (`testing/` - 15 PDFs):
- 5 claim form variations (`claim_form_test_1.pdf` to `claim_form_test_5.pdf`)
- 5 inspection report variations (`inspection_report_test_1.pdf` to `inspection_report_test_5.pdf`)
- 5 invoice variations (`invoice_test_1.pdf` to `invoice_test_5.pdf`)

## Setup

### One-Time Setup: Create Vector Database
```bash
# IMPORTANT: Run this FIRST before classifying documents
python chroma_dbmaker.py
```
This creates embeddings of the 3 reference documents in ChromaDB at `../shared/chroma_store/`
Collection name: `insurance_reference_docs`

**Note**: If you run this script multiple times, it will ask if you want to keep or recreate the collection.

### Running the Classifier

#### CLI Version
```bash
python main.py
```
Classifies all test documents and displays results with similarity scores.

#### Web UI Version
```bash
# Start Flask web server
python app.py

# Then open your browser to http://localhost:5004
# Upload documents to classify them automatically
```

## How It Works
1. **Reference Embeddings**: Reference PDFs are converted to vector embeddings
2. **Test Document Processing**: Test PDFs are extracted and embedded
3. **Similarity Matching**: Cosine similarity search finds the closest reference document
4. **Classification**: Document is classified based on highest similarity score
5. **Confidence Score**: Returns similarity percentage for transparency

## Example Output
```
Document: invoice_test_2.pdf
Classification: invoice_vector.pdf
Confidence: 94.5%
```

## Database
- **Collection Name**: `insurance_reference_docs`
- **Location**: `../shared/chroma_store/` (shared directory with Statement 4, separate collection)
- **Embedding Model**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Safety**: Script prevents accidental overwrites by prompting before recreating

## Technologies Used
- **LangChain**: Document processing and RAG pipeline
- **ChromaDB**: Vector database for reference document embeddings
- **HuggingFace Embeddings**: Sentence transformer model
- **PyPDFLoader**: PDF text extraction
- **Flask**: Web interface (for app.py)
