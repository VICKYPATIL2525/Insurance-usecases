# Statement 1: Policy Summary Assistant

## Description
Build a GenAI tool that can read lengthy insurance policy documents and generate a concise, plain-language summary of coverage, exclusions, and limits.

## Files
- `statement_1.py` - Original implementation
- `statement_1_optimized.py` - Optimized version with parallel batch processing (3-5x faster)

## Required Data
**IMPORTANT**: You need to add the insurance policy PDF file:
- Create a file named `health_insurance_document.pdf` in the `data/` folder
- The PDF should contain an insurance policy document to summarize

## How to Run
1. Place your insurance policy PDF in `data/health_insurance_document.pdf`
2. Run: `python statement_1.py` (original version)
   OR
   Run: `python statement_1_optimized.py` (faster version)

## Output
The script will generate a plain-language summary (under 200 words) covering:
- What is covered
- What is NOT covered (exclusions)
- Important limits, waiting periods, and conditions
