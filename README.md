# Insurance GenAI Solutions

This project contains 6 GenAI-powered insurance tools demonstrating various applications of LLMs in the insurance domain.

## Project Structure

```
vm_problem_statements/
├── statement_1_policy_summary/        # Policy document summarization
├── statement_2_claims_normalizer/     # Claims text to structured data
├── statement_4_quote_comparison/      # Insurance quote comparison chatbot
├── statement_5_underwriting/          # Risk assessment assistant
├── statement_6_document_classifier/   # Document classification system
├── shared/                            # Shared resources (vector database)
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
# Add your PDF to data/health_insurance_document.pdf first!
python statement_1_optimized.py
```

### Statement 2: Claims Normalizer
```bash
cd statement_2_claims_normalizer
python statement_2_batch.py
```

### Statement 4: Quote Comparison
```bash
cd statement_4_quote_comparison
python statement_4_dbmaker.py  # First time setup
python statement_4_main.py     # Run chatbot
```

### Statement 5: Underwriting Assistant
```bash
cd statement_5_underwriting
python statement_5_main.py
```

### Statement 6: Document Classifier
```bash
cd statement_6_document_classifier
python statement_6_dbmaker.py  # First time setup
python statement_6_main.py     # Classify documents
```

## Technologies Used

- **LangChain**: Framework for LLM applications
- **Azure OpenAI**: GPT-4 models for text generation
- **ChromaDB**: Vector database for embeddings
- **HuggingFace**: Sentence transformers for embeddings
- **PyPDF**: PDF document processing
- **Pydantic**: Data validation and structured outputs

## Important Notes

- **Statement 1** requires a PDF file that is NOT included in the repository
- **Statements 4 & 6** share the same vector database (`shared/chroma_store/`)
- All scripts use relative paths and should be run from their respective directories
- Make sure `.env` file is configured before running any scripts

## License

This is a demo project for educational purposes.
