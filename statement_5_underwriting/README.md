# Statement 5: Underwriting Assistant

## Description
An AI-powered underwriting co-pilot that automates insurance risk assessment by analyzing multiple applicant documents simultaneously. The system ingests three PDF document types per applicant—**Applicant Data** (personal info, medical history), **Prior Claims** (historical claims records), and **External Reports** (third-party assessments)—then uses Azure OpenAI GPT-4 to synthesize this information into a comprehensive risk evaluation. Output is structured via Pydantic schema into: a **Risk Score** (0-100), **Risk Level** (LOW/MEDIUM/HIGH), plain-language **Risk Summary**, itemized **Key Risk Factors**, **Positive Indicators**, and **Underwriter Notes**. Includes 10 pre-loaded test applicants across all risk tiers (30 PDFs total) for validation. The Flask web UI allows underwriters to upload new applicant documents and receive instant, consistent risk assessments—augmenting human judgment with AI-powered analysis.

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
