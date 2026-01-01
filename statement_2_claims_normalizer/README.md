# Statement 2: Claims Description Normalizer

## Description
An NLP-based claims processing tool that converts unstructured, free-text claim notes (written by adjusters or customers in varying styles) into standardized, structured JSON data. Using Azure OpenAI GPT-4 with Pydantic schema validation, the system extracts three critical fields from each claim: **loss_type** (e.g., Accident, Theft, Water Damage), **severity** (Low/Medium/High), and **affected_asset** (e.g., Vehicle, Property, Electronics). Supports both single-claim interactive processing and batch processing via CSV files containing multiple claims. The Flask web UI enables insurance staff to paste claim descriptions or upload CSV batches and receive consistent, machine-readable output—eliminating manual data entry and ensuring uniform claim categorization for downstream processing.

## Files
- `main_single_processing.py` - Single claim processing
- `main_batch_processing.py` - Batch processing from CSV file
- `app.py` - Flask web UI for interactive claim normalization
- `data/statement_2_claims.csv` - Sample claims data (19 claims)
- `templates/claims_normalizer.html` - Web interface HTML
- `output/` - Folder for generated JSON files
- `uploads/` - Folder for user-uploaded CSV files (web UI)

## Data Files
- `data/statement_2_claims.csv` - Contains 19 sample insurance claims with free-text descriptions

## How to Run

### CLI Version - Single Claim
```bash
python main_single_processing.py
```
Enter a claim description when prompted, and get structured JSON output.

### CLI Version - Batch Processing
```bash
python main_batch_processing.py
```
Processes all claims from `data/statement_2_claims.csv` and saves JSON files to `output/` folder.

### Web UI Version
```bash
# Start Flask web server
python app.py

# Then open your browser to http://localhost:5001
# Enter claims manually or upload CSV files
```

## Output Format
The system extracts structured data using Pydantic schema:
```json
{
  "loss_type": "Accident",
  "severity": "High",
  "affected_asset": "Vehicle"
}
```

## Technologies Used
- **LangChain**: LLM orchestration
- **Azure OpenAI GPT-4**: Text understanding and extraction
- **Pydantic**: Structured output validation (ClaimSchema)
- **Flask**: Web interface (for app.py)
