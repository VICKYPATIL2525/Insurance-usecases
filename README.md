# Insurance GenAI Solutions

A comprehensive suite of 5 GenAI-powered tools that automate and enhance core insurance operations using Large Language Models (Azure OpenAI GPT-4), vector databases (ChromaDB), and RAG (Retrieval-Augmented Generation) pipelines. Each tool addresses a specific insurance workflow—from policy document summarization and claims normalization to quote comparison, underwriting risk assessment, and document classification—demonstrating practical enterprise AI applications with both CLI and web interfaces.

## What This Project Does

| Tool | Problem Solved | AI Technique |
|------|----------------|--------------|
| **Policy Summary** | Condenses 50+ page insurance policies into plain-language summaries | Document chunking + LLM summarization |
| **Claims Normalizer** | Converts unstructured adjuster notes into structured JSON | LLM extraction with Pydantic validation |
| **Quote Comparison** | Answers natural language questions about insurance plans | RAG with two-LLM classification system |
| **Underwriting Assistant** | Automates risk scoring from applicant documents | Multi-document LLM analysis |
| **Document Classifier** | Categorizes incoming documents by type | Embedding similarity matching |

## Project Structure

```
vm_problem_statemets/
├── statement_1_policy_summary/        # Policy document summarization
│   ├── app.py                        # Flask web UI
│   ├── main.py                       # Original implementation
│   ├── main_optimized.py             # Optimized version (3-5x faster)
│   ├── data/                         # Sample insurance PDF
│   ├── templates/                    # Web interface HTML
│   └── uploads/                      # User-uploaded PDFs
│
├── statement_2_claims_normalizer/     # Claims text to structured data
│   ├── app.py                        # Flask web UI
│   ├── main_single_processing.py     # Single claim processing
│   ├── main_batch_processing.py      # CSV batch processing
│   ├── data/                         # Sample claims CSV
│   ├── output/                       # Generated JSON results
│   ├── templates/                    # Web interface HTML
│   └── uploads/                      # User-uploaded files
│
├── statement_4_quote_comparison/      # Insurance quote comparison chatbot
│   ├── app.py                        # Flask web UI
│   ├── main.py                       # Interactive chatbot
│   ├── chroma_db_maker.py            # Vector DB creation
│   ├── data/                         # Insurance plans JSON
│   └── templates/                    # Web interface HTML
│
├── statement_5_underwriting/          # Risk assessment assistant
│   ├── app.py                        # Flask web UI
│   ├── main.py                       # Underwriting analysis
│   ├── data/                         # 10 applicant folders (30 PDFs)
│   ├── templates/                    # Web interface HTML
│   └── uploads/                      # User-uploaded documents
│
├── statement_6_document_classifier/   # Document classification system
│   ├── app.py                        # Flask web UI
│   ├── main.py                       # Classification logic
│   ├── chroma_dbmaker.py             # Vector DB creation
│   ├── data/                         # Reference and test PDFs
│   ├── templates/                    # Web interface HTML
│   └── uploads/                      # User-uploaded documents
│
├── shared/                            # Shared resources
│   └── chroma_store/                 # Vector database (ChromaDB)
│
├── .env                               # Environment variables (API keys)
├── .gitignore                         # Git ignore file
└── requirments.txt                    # Python dependencies
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirments.txt
   ```

2. **Configure Environment Variables**
   - Copy and configure `.env` file with your Azure OpenAI credentials
   - Required variables:
     - `AZURE_OPENAI_ENDPOINT`
     - `AZURE_OPENAI_VERSION`
     - `OPENAI_API_KEY`

3. **Run Individual Statements**
   - Navigate to each statement folder
   - Follow the README.md in that folder

## Quick Start Guide

### Statement 1: Policy Summary Assistant
```bash
cd statement_1_policy_summary
# CLI version (sample PDF included)
python main.py               # Original version
python main_optimized.py     # Optimized version (3-5x faster)

# Web UI version
python app.py                # Access at http://localhost:5000
```

### Statement 2: Claims Normalizer
```bash
cd statement_2_claims_normalizer
# CLI version
python main_single_processing.py  # Process single claims
python main_batch_processing.py   # Process CSV file

# Web UI version
python app.py                     # Access at http://localhost:5001
```

### Statement 4: Quote Comparison
```bash
cd statement_4_quote_comparison
# CLI version
python chroma_db_maker.py    # First time setup (creates vector DB)
python main.py               # Run interactive chatbot

# Web UI version
python app.py                # Access at http://localhost:5002
```

### Statement 5: Underwriting Assistant
```bash
cd statement_5_underwriting
# CLI version (edit line 228 in main.py to change applicant)
python main.py

# Web UI version
python app.py                # Access at http://localhost:5003
```

### Statement 6: Document Classifier
```bash
cd statement_6_document_classifier
# CLI version
python chroma_dbmaker.py     # First time setup (creates vector DB)
python main.py               # Classify test documents

# Web UI version
python app.py                # Access at http://localhost:5004
```

## Technologies Used

- **LangChain**: Framework for LLM applications
- **Azure OpenAI**: GPT-4 models for text generation
- **ChromaDB**: Vector database for embeddings
- **HuggingFace**: Sentence transformers for embeddings
- **PyPDF**: PDF document processing
- **Pydantic**: Data validation and structured outputs

## ChromaDB Collections

The shared vector database (`shared/chroma_store/`) contains two separate collections:

| Collection Name | Used By | Purpose |
|----------------|---------|---------|
| `insurance_quotes` | Statement 4 | Insurance plan comparison data |
| `insurance_reference_docs` | Statement 6 | Document classification references |

**Important Notes:**
- Collections are independent - you can run db maker scripts in any order
- Running a db maker script multiple times will prompt you to skip or recreate
- Safe to run both statements simultaneously without conflicts

## Important Notes

- **Statement 1** includes a sample PDF (`health_insurance_document.pdf`) in the data folder
- **Statement 3** is not present in this codebase (only Statements 1, 2, 4, 5, 6 are implemented)
- **Statements 4 & 6** share the same vector database directory (`shared/chroma_store/`) but use separate collections
- **Each statement has both CLI and Web UI versions** - run `app.py` for web interface
- All scripts use relative paths and should be run from their respective directories
- Make sure `.env` file is configured before running any scripts
- Vector database must be created first for Statements 4 & 6 (run the `chroma_db_maker.py` or `chroma_dbmaker.py` scripts)
- DB maker scripts have safety checks to prevent accidental data overwrites

## License

This is a demo project for educational purposes.
