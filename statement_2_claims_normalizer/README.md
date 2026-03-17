# Statement 2: Claims Description Normalizer

## Description
An NLP-based claims processing tool that converts unstructured, free-text claim notes (written by adjusters or customers in varying styles) into standardized, structured JSON data. Using Azure OpenAI GPT-4.1-mini with TypedDict schema validation, the system extracts three critical fields from each claim: **loss_type** (e.g., Accident, Theft, Water Damage), **severity** (Low/Medium/High), and **affected_asset** (e.g., Vehicle, Property, Electronics). Supports both single-claim interactive processing and batch processing via CSV files containing multiple claims. The Flask web UI enables insurance staff to paste claim descriptions or upload CSV batches and receive consistent, machine-readable output—eliminating manual data entry and ensuring uniform claim categorization for downstream processing.

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

### CLI Version - Single Claim (Hardcoded Example)
```bash
python main_single_processing.py
```
Processes a hardcoded example claim (vehicle accident) and saves the structured JSON output to `output/`. To test a different claim, edit the `claim_text` variable in the script.

### CLI Version - Interactive Menu
```bash
python main_batch_processing.py
```
Shows a menu with 3 options:
1. **Single Claim Processing** — processes a hardcoded example claim
2. **Batch Processing (from CSV)** — reads all claims from `data/statement_2_claims.csv`, processes each one, and saves individual + combined JSON files to `output/`
3. **Exit**

### Web UI Version
```bash
# Start Flask web server
python app.py

# Then open your browser to http://localhost:5001
# Enter claims manually or upload CSV files
```

### API Endpoints (app.py)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/normalize-single` | Normalize a single claim (JSON body: `claim_text`) |
| POST | `/normalize-batch` | Upload CSV file with multiple claims |

## Output Format
The system extracts structured data using TypedDict schema (`ClaimSchema`):
```json
{
  "loss_type": "Accident",
  "severity": "High",
  "affected_asset": "Vehicle"
}
```

Batch processing also saves a combined JSON file with all results timestamped (e.g., `batch_results_17-03-2026_02-30-00_PM.json`).

## Technologies Used
- **LangChain**: LLM orchestration with `with_structured_output()` for schema enforcement
- **Azure OpenAI (gpt-4.1-mini)**: Text understanding and extraction
- **TypedDict**: Structured output schema (`ClaimSchema` with `loss_type`, `severity`, `affected_asset`)
- **Flask**: Web interface (for app.py)
