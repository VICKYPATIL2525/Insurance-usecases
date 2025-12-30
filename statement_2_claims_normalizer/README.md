# Statement 2: Claims Description Normalizer

## Description
Develop an AI model that converts raw claim notes (free-text from adjusters or customers) into structured data—detecting loss type, severity, and affected asset.

## Files
- `statement_2.py` - Single claim processing
- `statement_2_batch.py` - Batch processing from CSV file

## Data Files
- `data/statement_2_claims.csv` - Sample claims data for batch processing

## Output
- `output/` - JSON files with structured claim data

## How to Run
### Single Claim
```bash
python statement_2.py
```

### Batch Processing
```bash
python statement_2_batch.py
```

## Output Format
```json
{
  "loss_type": "Accident",
  "severity": "High",
  "affected_asset": "Vehicle"
}
```
