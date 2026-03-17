# Statement 1: Policy Summary Assistant

## Description
An LLM-powered document summarization tool that transforms lengthy insurance policy PDFs (often 50+ pages of legal language) into concise, customer-friendly summaries under 200 words. The system uses LangChain's RecursiveCharacterTextSplitter to chunk documents into processable segments, then leverages Azure OpenAI GPT-4.1-mini to extract and synthesize key policy information—coverage details, exclusions, limits, waiting periods, and special conditions. Includes an optimized version with parallel batch processing that achieves 3-5x faster processing for large documents. Both CLI and Flask web interfaces allow users to upload any insurance PDF and receive instant plain-language summaries.

## Files
- `cli-v1.py` - Original implementation
- `cli-v2-optimized.py` - Optimized version with parallel batch processing (3-5x faster)
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
python cli-v1.py

# Optimized version (3-5x faster with parallel processing)
python cli-v2-optimized.py
```

### Web UI Version
```bash
# Start Flask web server
python app.py

# Then open your browser to http://localhost:5000
# Upload any insurance policy PDF and get instant summaries
```

The web UI features **real-time progress tracking** via Server-Sent Events (SSE). Processing runs in a background thread, and the frontend receives live updates at each stage (extraction, preprocessing, chunking, chunk summarization, final summary).

### API Endpoints (app.py)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF, returns `task_id` for progress tracking |
| GET | `/progress/<task_id>` | SSE stream for real-time progress updates |
| POST | `/summarize` | Summarize a PDF by server path (JSON body: `pdf_path`) |

- Max upload size: 16MB

## Output
The script will generate a plain-language summary (under 200 words) covering:
- What is covered
- What is NOT covered (exclusions)
- Important limits, waiting periods, and conditions

## Technologies Used
- **LangChain**: Document processing and LLM orchestration
- **PyPDFLoader**: PDF text extraction
- **RecursiveCharacterTextSplitter**: Document chunking (chunk_size=3000, chunk_overlap=150)
- **Azure OpenAI (gpt-4.1-mini)**: Summary generation
- **Flask**: Web interface with SSE progress tracking (for app.py)
- **Threading**: Background processing for non-blocking web requests
