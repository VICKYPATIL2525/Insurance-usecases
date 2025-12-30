# Statement 5: Underwriting Assistant

## Description
A GenAI-powered co-pilot that helps underwriters assess risk by summarizing applicant data, prior claims, and external reports.

## Files
- `statement_5_main.py` - Main underwriting analysis script

## Data Structure
The `data/` folder contains 9 applicant folders with risk levels:
- `Ananya_Sharma_LOW/`
- `Arjun_Nair_LOW/`
- `Bharat_Mishra_HIGH/`
- `Kamala_Venkatesh_HIGH/`
- `Mohammed_Irfan_Khan_MEDIUM/`
- `Neha_Kapoor_LOW/`
- `Prakash_Choudhary_HIGH/`
- `Rajesh_Kumar_Singh_MEDIUM/`
- `Sunita_Devi_Agarwal_MEDIUM/`
- `Vikram_Patel_LOW/`

Each folder contains PDF documents for that applicant.

## How to Run
Edit `statement_5_main.py` line 228 to change the applicant folder, then:
```bash
python statement_5_main.py
```

## Output
Structured risk assessment with:
- Risk score (0-100)
- Risk level (LOW/MEDIUM/HIGH)
- Risk summary
- Key risk factors
- Positive indicators
- Underwriter notes
