'''Flask Web Application for Document Classifier
Wraps the main.py logic for PDF classification into a web interface'''

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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

# ---------------- Configuration (from main.py) ----------------
CHROMA_DB_PATH = "../shared/chroma_store"
COLLECTION_NAME = "insurance_reference_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- Initialize Embeddings and Vector DB ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = Chroma(
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

# ---------------- Core Logic (from main.py) ----------------
def load_pdf_text(pdf_path):
    """Loads a PDF file and extracts all text from it."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

def classify_document(pdf_path, top_k=3):
    """
    Classify a single PDF document against reference documents.

    Returns:
        dict with classification results
    """
    # Extract text from PDF
    query_text = load_pdf_text(pdf_path)

    # Search vector database for similar documents
    results = vectordb.similarity_search_with_score(query_text, k=top_k)

    # Process results
    classifications = []
    best_match = None
    best_score = 0

    for doc, score in results:
        similarity = (1 - score) * 100
        source = doc.metadata.get("source", "unknown")

        classifications.append({
            "source": source,
            "similarity": round(similarity, 2)
        })

        if similarity > best_score:
            best_score = similarity
            best_match = source

    return {
        "best_match": best_match,
        "confidence": round(best_score, 2),
        "all_matches": classifications
    }

# ---------------- Flask Routes ----------------
@app.route('/', methods=['GET'])
def home():
    return render_template('document_classifier.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/classify-single', methods=['POST'])
def classify_single():
    """Classify a single PDF document"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Classify the document
        result = classify_document(filepath)

        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "success": True,
            "filename": filename,
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/classify-batch', methods=['POST'])
def classify_batch():
    """Classify multiple PDF documents"""
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

        # Create a unique folder for this batch
        import time
        batch_id = str(int(time.time() * 1000))
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
        os.makedirs(batch_folder, exist_ok=True)

        results = []

        # Process each file
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(batch_folder, filename)
            file.save(filepath)

            try:
                # Classify the document
                classification = classify_document(filepath)

                results.append({
                    "filename": filename,
                    "success": True,
                    **classification
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e)
                })

        # Clean up uploaded files
        shutil.rmtree(batch_folder)

        return jsonify({
            "success": True,
            "total_files": len(files),
            "results": results
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
