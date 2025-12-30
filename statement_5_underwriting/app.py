'''Flask Web Application for Underwriting Assistant
Wraps the main.py logic for risk assessment into a web interface'''

import os
import re
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List
import shutil

# ---------------- Load env ----------------
load_dotenv()

# ---------------- Initialize Flask App ----------------
app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Pydantic Model (from main.py) ----------------
class UnderwritingAnalysis(BaseModel):
    """Structured output model for underwriting analysis."""
    risk_score: int = Field(..., ge=0, le=100, description="Risk score between 0 and 100")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    risk_summary: str = Field(..., description="2-3 sentence overview of the risk")
    key_risk_factors: List[str] = Field(..., description="List of key risk factors")
    positive_indicators: List[str] = Field(..., description="List of positive indicators")
    underwriter_notes: str = Field(..., description="Short actionable note for underwriter")

# ---------------- LLM Setup (from main.py) ----------------
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=1000,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30,
    max_retries=2,
)

UNDERWRITING_SYSTEM_PROMPT = """
You are an insurance underwriting co-pilot.

Your job is to assist a human underwriter by analyzing:
1. Applicant data
2. Prior claims history
3. External verification reports

You must:
- Summarize overall risk clearly
- Highlight key risk factors
- Highlight positive indicators
- Assign a risk score from 0 to 100
- Classify risk as LOW, MEDIUM, or HIGH
- Do NOT approve or reject the policy
- Do NOT invent facts
- Base conclusions ONLY on the provided data
"""

# ---------------- Core Logic (from main.py) ----------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF file."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

def preprocess_text(text: str) -> str:
    """Basic text preprocessing - remove extra whitespace."""
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

def read_all_pdfs_from_folder(folder_path: str) -> dict:
    """Read all PDF files from a folder and return their contents."""
    results = {}

    if not os.path.exists(folder_path):
        return results

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        return results

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            clean_text = preprocess_text(raw_text)
            results[pdf_file] = clean_text
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            results[pdf_file] = None

    return results

def generate_underwriting_summary(documents: dict, max_chars_per_doc: int = 5000) -> UnderwritingAnalysis:
    """Generate risk summary using LLM with structured output."""

    combined_text = ""

    for filename, content in documents.items():
        if content:
            truncated_content = content[:max_chars_per_doc]
            if len(content) > max_chars_per_doc:
                truncated_content += f"\n[... truncated {len(content) - max_chars_per_doc} characters ...]"
            combined_text += f"\n\n--- {filename} ---\n{truncated_content}"

    if not combined_text.strip():
        return UnderwritingAnalysis(
            risk_score=0,
            risk_level="UNKNOWN",
            risk_summary="No valid document content available for underwriting.",
            key_risk_factors=["No documents provided"],
            positive_indicators=[],
            underwriter_notes="Cannot proceed without documents."
        )

    structured_llm = llm.with_structured_output(UnderwritingAnalysis)

    messages = [
        SystemMessage(content=UNDERWRITING_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Below are the underwriting documents for a single applicant.

Analyze them carefully and provide a structured risk assessment.

DOCUMENTS:
{combined_text}
""")
    ]

    response = structured_llm.invoke(messages)
    return response

def format_underwriting_output(analysis: UnderwritingAnalysis) -> dict:
    """Format the structured analysis into a dictionary."""
    return {
        "risk_score": analysis.risk_score,
        "risk_level": analysis.risk_level,
        "risk_summary": analysis.risk_summary,
        "key_risk_factors": analysis.key_risk_factors,
        "positive_indicators": analysis.positive_indicators,
        "underwriter_notes": analysis.underwriter_notes
    }

# ---------------- Flask Routes ----------------
@app.route('/', methods=['GET'])
def home():
    return render_template('underwriting.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_risk():
    try:
        if 'files' not in request.files:
            return jsonify({"success": False, "error": "No files uploaded"}), 400

        files = request.files.getlist('files')

        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No files selected"}), 400

        # Validate all files
        for file in files:
            if file.filename == '':
                return jsonify({"success": False, "error": "Empty filename found"}), 400
            if not allowed_file(file.filename):
                return jsonify({"success": False, "error": f"File {file.filename} is not a PDF"}), 400

        # Create a unique folder for this analysis
        import time
        analysis_id = str(int(time.time() * 1000))
        analysis_folder = os.path.join(app.config['UPLOAD_FOLDER'], analysis_id)
        os.makedirs(analysis_folder, exist_ok=True)

        # Save all uploaded files
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(analysis_folder, filename)
            file.save(filepath)

        # Process PDFs
        documents = read_all_pdfs_from_folder(analysis_folder)

        if not documents:
            shutil.rmtree(analysis_folder)
            return jsonify({"success": False, "error": "Failed to extract text from PDFs"}), 400

        # Generate risk analysis
        analysis = generate_underwriting_summary(documents)
        result = format_underwriting_output(analysis)

        # Clean up uploaded files
        shutil.rmtree(analysis_folder)

        return jsonify({
            "success": True,
            "data": {
                **result,
                "documents_analyzed": len(documents)
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
