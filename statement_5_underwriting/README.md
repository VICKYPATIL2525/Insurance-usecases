# Statement 5: Underwriting Assistant

## Description
A GenAI-powered co-pilot that helps underwriters assess risk by summarizing applicant data, prior claims, and external reports.

## Files
- `main.py` - Main underwriting analysis script (CLI version)
- `app.py` - Flask web UI for risk assessment
- `templates/underwriting.html` - Web interface HTML
- `uploads/` - Folder for user-uploaded documents (web UI)
- `data/` - Contains 10 applicant folders with sample data

## Data Structure
The `data/` folder contains **10 applicant folders** (30 PDFs total) with pre-classified risk levels:

### LOW Risk (4 applicants)
- `Ananya_Sharma_LOW/`
- `Arjun_Nair_LOW/`
- `Neha_Kapoor_LOW/`
- `Vikram_Patel_LOW/`

### MEDIUM Risk (3 applicants)
- `Mohammed_Irfan_Khan_MEDIUM/`
- `Rajesh_Kumar_Singh_MEDIUM/`
- `Sunita_Devi_Agarwal_MEDIUM/`

### HIGH Risk (3 applicants)
- `Bharat_Mishra_HIGH/`
- `Kamala_Venkatesh_HIGH/`
- `Prakash_Choudhary_HIGH/`

Each applicant folder contains 3 PDFs:
1. `1_Applicant_Data.pdf` - Personal information, medical history
2. `2_Prior_Claims.pdf` - Claims history
3. `3_External_Reports.pdf` - Third-party reports and assessments

## How to Run

### CLI Version
```bash
# Edit main.py line 228 to change the applicant folder
python main.py
```

Example: Change line 228 from:
```python
applicant_folder = "Vikram_Patel_LOW"
```
to:
```python
applicant_folder = "Bharat_Mishra_HIGH"
```

### Web UI Version
```bash
# Start Flask web server
python app.py

# Then open your browser to http://localhost:5003
# Upload applicant documents and get instant risk assessment
```

## Output
Structured risk assessment using Pydantic `UnderwritingAnalysis` schema:
- **Risk Score**: 0-100 numerical score
- **Risk Level**: LOW / MEDIUM / HIGH
- **Risk Summary**: Plain-language explanation
- **Key Risk Factors**: List of concerns
- **Positive Indicators**: List of favorable factors
- **Underwriter Notes**: Additional observations

## Technologies Used
- **LangChain**: Document processing and LLM orchestration
- **PyPDFLoader**: PDF text extraction
- **Azure OpenAI GPT-4**: Risk assessment analysis
- **Pydantic**: Structured output validation (UnderwritingAnalysis schema)
- **Flask**: Web interface (for app.py)
