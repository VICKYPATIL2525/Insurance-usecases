# Statement 4: Quote Comparison Chatbot

## Description
A RAG-powered (Retrieval-Augmented Generation) conversational chatbot that helps customers and agents compare insurance quotes through natural language questions. The system employs a **two-LLM architecture**: a Classification LLM first determines whether the user's question is a new comparison request (2-3 premiums), a follow-up question, or invalid, then an Answer LLM generates responses using semantically retrieved context from ChromaDB.

Insurance plan data (premiums, deductibles, coverage limits, exclusions, waiting periods) is embedded using HuggingFace sentence-transformers and stored in a vector database for efficient similarity search.

**Workflow:** Users must first compare 2-3 plans by providing premium amounts (e.g., "compare 18000, 22500, 28000"), then they can ask follow-up questions like "Which has no deductible?" or "Which is best for a family of 4?" The chatbot maintains conversation context to answer detailed comparison questions.

## Files
- `chroma_db_maker.py` - Creates vector database from insurance plans
- `main.py` - Interactive chatbot (CLI version)
- `app.py` - Flask web UI for chat interface
- `data/statement_4_data.json` - Insurance plan data (5+ plans with full details)
- `data/statement_4_questions.json` - Sample questions for testing
- `templates/quote_comparison.html` - Web interface HTML

## Data Files
- `data/statement_4_data.json` - Contains 10 insurance plans with:
  - Premium amounts: ₹16,500, ₹18,000, ₹19,500, ₹21,000, ₹22,500, ₹25,000, ₹27,500, ₹28,000, ₹30,000, ₹35,000
  - Sum Insured: Ranging from ₹8 Lakhs to ₹25 Lakhs
  - Deductibles: Some plans have no deductible, others range from ₹10,000 to ₹25,000
  - Coverage details: Hospitalization, ICU, day-care procedures, maternity (select plans)
  - Exclusions: OPD, cosmetic procedures, dental (unless accidental)
  - Waiting periods: 30 days initial, 24-48 months for pre-existing conditions
  - Family coverage: 3-5 members depending on plan

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

### Initial Comparison (Required First Step)
Start by comparing 2-3 plans using their premium amounts:
- "compare 18000, 22500, 28000"
- "18000 vs 22500"
- "compare plans 18000 and 22500 for family of 4"
- "show me 20000, 25000, 30000"

### Follow-up Questions (After Initial Comparison)
Once you've compared plans, you can ask follow-up questions:
- "which one has no deductible?"
- "which is best for family of 4?"
- "is the extra cost worth it?"
- "what are the waiting periods?"
- "which plan covers pre-existing conditions?"
- "which is cheaper?"
- "what's the difference between them?"

**Note:** The chatbot requires you to first provide 2-3 premium amounts to compare. Follow-up questions will only work after you've made an initial comparison.

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
