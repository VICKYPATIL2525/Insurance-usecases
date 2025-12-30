# Statement 4: Quote Comparison Chatbot

## Description
A conversational assistant that compares multiple insurance quotes and explains differences in coverage, premium, and deductible in simple terms.

## Files
- `statement_4_dbmaker.py` - Creates vector database from insurance plans
- `statement_4_main.py` - Interactive chatbot

## Data Files
- `data/statement_4_data.json` - Insurance plan data
- `data/statement_4_questions.json` - Sample questions

## Setup
1. First, create the vector database:
   ```bash
   python statement_4_dbmaker.py
   ```

2. Then run the chatbot:
   ```bash
   python statement_4_main.py
   ```

## Usage Examples
- "compare 18000, 22500, 28000"
- "which one has no deductible?"
- "which is best for family of 4?"
- "is the extra cost worth it?"

## Database
Uses shared vector database at `../shared/chroma_store/`
