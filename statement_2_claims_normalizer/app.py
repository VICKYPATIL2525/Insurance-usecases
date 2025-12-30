'''Flask Web Application for Claims Description Normalizer
Wraps the main_batch_processing.py logic into a web interface'''

import os
import json
import csv
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import TypedDict, Optional

# ---------------- Load env ----------------
load_dotenv()

# ---------------- Initialize Flask App ----------------
app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Schema definition ----------------
class ClaimSchema(TypedDict):
    loss_type: Optional[str]
    severity: Optional[str]
    affected_asset: Optional[str]

# ---------------- Azure OpenAI LLM ----------------
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=200,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Wrap LLM to enforce structured output
structured_llm = llm.with_structured_output(ClaimSchema)

# ---------------- Core Logic (from main_batch_processing.py) ----------------
def normalize_claim_text(claim_text: str) -> ClaimSchema:
    """
    Processes a raw claim description and extracts structured data.

    Args:
        claim_text: Unstructured claim description text

    Returns:
        ClaimSchema: Dictionary with loss_type, severity, and affected_asset
    """
    prompt = f"""
You are an insurance claims expert.

From the following claim description, extract:
- loss_type (Accident, Theft, Fire, Water Damage, Health, etc.)
- severity (Low, Medium, High)
- affected_asset (Vehicle, Property, Health, Electronics, etc.)

Rules:
- Return ONLY structured data
- If information is missing, return null
- Do NOT add assumptions

Claim description:
{claim_text}
"""
    return structured_llm.invoke(prompt)

# ---------------- Flask Routes ----------------
@app.route('/', methods=['GET'])
def home():
    return render_template('claims_normalizer.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/normalize-single', methods=['POST'])
def normalize_single():
    """Process a single claim text"""
    try:
        data = request.get_json()

        if not data or 'claim_text' not in data:
            return jsonify({"success": False, "error": "claim_text is required"}), 400

        claim_text = data['claim_text'].strip()

        if not claim_text:
            return jsonify({"success": False, "error": "claim_text cannot be empty"}), 400

        # Normalize the claim
        result = normalize_claim_text(claim_text)

        # Save to JSON file with timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        filename = f"claim_output_{timestamp}.json"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)

        output_data = {
            "claim_text": claim_text,
            **result
        }

        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

        return jsonify({
            "success": True,
            "data": result,
            "saved_to": filename
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/normalize-batch', methods=['POST'])
def normalize_batch():
    """Process multiple claims from CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Only CSV files are allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read CSV file
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            claims = list(reader)

        # Validate CSV structure
        if not claims or 'claim_id' not in claims[0] or 'claim_text' not in claims[0]:
            os.remove(filepath)
            return jsonify({
                "success": False,
                "error": "CSV must contain 'claim_id' and 'claim_text' columns"
            }), 400

        results = []
        errors = []

        # Process each claim
        for idx, claim in enumerate(claims, 1):
            claim_id = claim['claim_id']
            claim_text = claim['claim_text']

            try:
                # Normalize the claim
                result = normalize_claim_text(claim_text)

                result_with_id = {
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    **result
                }

                results.append(result_with_id)

                # Save individual JSON file
                individual_filename = f"{claim_id}.json"
                individual_filepath = os.path.join(app.config['OUTPUT_FOLDER'], individual_filename)
                with open(individual_filepath, "w") as f:
                    json.dump(result_with_id, f, indent=2)

            except Exception as e:
                errors.append({
                    "claim_id": claim_id,
                    "error": str(e)
                })

        # Save combined results
        timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        combined_filename = f"batch_results_{timestamp}.json"
        combined_filepath = os.path.join(app.config['OUTPUT_FOLDER'], combined_filename)

        with open(combined_filepath, "w") as f:
            json.dump(results, f, indent=2)

        # Clean up uploaded CSV
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "success": True,
            "data": {
                "total_claims": len(claims),
                "processed": len(results),
                "errors": len(errors),
                "results": results,
                "error_details": errors,
                "saved_to": combined_filename
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
