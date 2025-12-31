# Statement 4: Quote Comparison Chatbot

## Description
A conversational assistant that compares multiple insurance quotes and explains differences in coverage, premium, and deductible in simple terms.

## Files
- `chroma_db_maker.py` - Creates vector database from insurance plans
- `main.py` - Interactive chatbot (CLI version)
- `app.py` - Flask web UI for chat interface
- `data/statement_4_data.json` - Insurance plan data (5+ plans with full details)
- `data/statement_4_questions.json` - Sample questions for testing
- `templates/quote_comparison.html` - Web interface HTML

## Data Files
- `data/statement_4_data.json` - Contains 5+ insurance plans with:
  - Premium amounts
  - Deductibles
  - Coverage details
  - Exclusions
  - Waiting periods
  - Additional features

## Setup

### One-Time Setup: Create Vector Database
```bash
# IMPORTANT: Run this FIRST before using the chatbot
python chroma_db_maker.py
```
This creates embeddings of insurance plans in the shared ChromaDB at `../shared/chroma_store/`
Collection name: `insurance_quotes`

**Note**: If you run this script multiple times, it will ask if you want to keep or recreate the collection.

### Running the Chatbot

#### CLI Version
```bash
python main.py
```

#### Web UI Version
```bash
# Start Flask web server
python app.py

# Then open your browser to http://localhost:5002
```

## Usage Examples
Try these questions:
- "compare 18000, 22500, 28000"
- "which one has no deductible?"
- "which is best for family of 4?"
- "is the extra cost worth it?"
- "what are the waiting periods?"
- "which plan covers pre-existing conditions?"

## How It Works
1. **Two-LLM System**:
   - Classification LLM: Determines if the question needs retrieval from the database
   - Answer LLM: Generates responses using retrieved context
2. **RAG (Retrieval-Augmented Generation)**: Uses ChromaDB for semantic search
3. **Embedding Model**: HuggingFace sentence-transformers/all-MiniLM-L6-v2

## Database
Uses shared vector database at `../shared/chroma_store/`

## Technologies Used
- **LangChain**: RAG pipeline orchestration
- **ChromaDB**: Vector database for insurance plan embeddings
- **HuggingFace Embeddings**: Sentence transformer model
- **Azure OpenAI GPT-4**: Question classification and answer generation
- **Flask**: Web interface (for app.py)
