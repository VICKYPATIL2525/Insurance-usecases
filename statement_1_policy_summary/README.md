# Statement 1: Policy Summary Assistant

## Description
An LLM-powered document summarization tool that transforms lengthy insurance policy PDFs (often 50+ pages of legal language) into concise, customer-friendly summaries under 200 words. The system uses LangChain's RecursiveCharacterTextSplitter to chunk documents into processable segments, then leverages Azure OpenAI GPT-4 to extract and synthesize key policy information—coverage details, exclusions, limits, waiting periods, and special conditions. Includes an optimized version with parallel batch processing that achieves 3-5x faster processing for large documents. Both CLI and Flask web interfaces allow users to upload any insurance PDF and receive instant plain-language summaries.

## Files
- `main.py` - Original implementation
- `main_optimized.py` - Optimized version with parallel batch processing (3-5x faster)
- `app.py` - Flask web UI for uploading and summarizing PDFs
- `data/health_insurance_document.pdf` - Sample insurance policy document (172 KB)
- `templates/index.html` - Web interface HTML
- `uploads/` - Folder for user-uploaded PDFs (web UI)

## Data Files
A sample insurance policy PDF is included:
- `data/health_insurance_document.pdf` - Sample health insurance policy document

You can also use your own PDF files by:
- Replacing the sample PDF in the `data/` folder, OR
- Using the web UI to upload your own PDF

## How to Run

### CLI Version
```bash
# Original version
python main.py

# Optimized version (3-5x faster with parallel processing)
python main_optimized.py
```

### Web UI Version
```bash
# Start Flask web server
python app.py

# Then open your browser to http://localhost:5000
# Upload any insurance policy PDF and get instant summaries
```

## Output
The script will generate a plain-language summary (under 200 words) covering:
- What is covered
- What is NOT covered (exclusions)
- Important limits, waiting periods, and conditions

## Technologies Used
- **LangChain**: Document processing and LLM orchestration
- **PyPDFLoader**: PDF text extraction
- **RecursiveCharacterTextSplitter**: Document chunking
- **Azure OpenAI GPT-4**: Summary generation
- **Flask**: Web interface (for app.py)
