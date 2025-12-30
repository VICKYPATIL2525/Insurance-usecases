'''Flask Web Application for Policy Summary Assistant
Wraps the main_optimized.py logic with parallel processing into a web interface'''

import os
import re
import time
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
from threading import Thread
import queue
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------- Load env ----------------
load_dotenv()

# ---------------- Initialize Flask App ----------------
app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Azure OpenAI LLM ----------------
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=400,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------- Extract PDF Text ----------------
def extract_text_from_pdf_langchain(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

# ---------------- Simple Lossless Preprocessing ----------------
def preprocess_text_basic(text: str) -> tuple[str, dict]:
    original_length = len(text)

    tab_count = text.count("\t")
    multi_space_count = len(re.findall(r" {2,}", text))
    blank_line_count = len(re.findall(r"\n\s*\n\s*\n+", text))

    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = text.strip()

    cleaned_length = len(text)

    stats = {
        "original_characters": original_length,
        "cleaned_characters": cleaned_length,
        "characters_removed": original_length - cleaned_length,
        "tabs_removed": tab_count,
        "multi_spaces_collapsed": multi_space_count,
        "extra_blank_lines_removed": blank_line_count,
        "reduction_percent": round(
            ((original_length - cleaned_length) / original_length) * 100, 2
        ) if original_length else 0.0
    }

    return text, stats

# ---------------- MAP: Summarize Chunks (OPTIMIZED WITH PARALLEL PROCESSING) ----------------
def summarize_chunks_parallel(chunks: list[str], progress_queue=None) -> list[str]:
    """
    ⚡ OPTIMIZED VERSION using LangChain's batch() method for parallel processing

    PERFORMANCE IMPACT:
    - Original: 10 chunks × 5 seconds = 50 seconds
    - Optimized: 10 chunks with max_concurrency=5 ≈ 10-15 seconds
    - Speedup: ~3-5x faster
    """

    system_prompt = (
        "You are an insurance domain expert. "
        "Summarize the following section of a policy document. "
        "Extract facts only. Do NOT add assumptions. "
        "Focus on coverage, exclusions, limits, waiting periods, and conditions."
    )

    # Build all messages upfront for batch processing
    all_messages = [
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=chunk)
        ]
        for chunk in chunks
    ]

    print(f"⚡ Processing {len(chunks)} chunks in parallel...")
    start_time = time.time()

    if progress_queue:
        progress_queue.put({"stage": "chunks", "progress": 0, "total": len(chunks), "percent": 25})

    # Process in batches to show progress
    max_concurrency = 10
    summaries = []

    for i in range(0, len(all_messages), max_concurrency):
        batch = all_messages[i:i + max_concurrency]

        responses = llm.batch(
            batch,
            config={"max_concurrency": max_concurrency}
        )

        summaries.extend([response.content for response in responses])

        # Update progress
        processed = min(i + max_concurrency, len(chunks))
        if progress_queue:
            # Calculate progress from 25% to 75% (50% range for chunks)
            # Start at 25% (after chunking) and add up to 50% based on chunk progress
            chunk_percent = 25 + int((processed / len(chunks)) * 50)
            progress_queue.put({
                "stage": "chunks",
                "progress": processed,
                "total": len(chunks),
                "percent": chunk_percent
            })
        print(f"✔ Processed {processed}/{len(chunks)} chunks")

    elapsed_time = time.time() - start_time
    print(f"✅ All {len(chunks)} chunks summarized in {elapsed_time:.2f} seconds")
    print(f"⚡ Average time per chunk: {elapsed_time/len(chunks):.2f} seconds")

    return summaries

# ---------------- REDUCE: Final Summary ----------------
def generate_final_summary(chunk_summaries: list[str], progress_queue=None) -> str:
    combined_summary = "\n".join(chunk_summaries)

    if progress_queue:
        progress_queue.put({"stage": "final", "progress": 0, "percent": 80})

    system_prompt = (
        "You are an insurance expert. "
        "Using the following section summaries, generate a concise, plain-language "
        "insurance policy summary.\n\n"
        "Clearly explain:\n"
        "1. What is covered\n"
        "2. What is NOT covered (exclusions)\n"
        "3. Important limits, waiting periods, and conditions\n\n"
        "For the most important points add the bullet points "
        "Keep it under 200 words and easy for a non-technical reader." \
        "add this line at the end This summary is for informational purposes only and does not replace the full policy document."

    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=combined_summary)
    ]

    if progress_queue:
        progress_queue.put({"stage": "final", "progress": 50, "percent": 90})

    response = llm.invoke(messages)

    if progress_queue:
        progress_queue.put({"stage": "final", "progress": 100, "percent": 100})

    return response.content

# ---------------- Process Policy Function ----------------
def process_policy(pdf_path: str, progress_queue=None) -> dict:
    # Extract text
    if progress_queue:
        progress_queue.put({"stage": "extract", "progress": 0, "percent": 5})

    pdf_text = extract_text_from_pdf_langchain(pdf_path)

    # Preprocess
    if progress_queue:
        progress_queue.put({"stage": "preprocess", "progress": 100, "percent": 15})

    clean_text, stats = preprocess_text_basic(pdf_text)

    # Chunking
    if progress_queue:
        progress_queue.put({"stage": "chunking", "progress": 100, "percent": 20})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=150
    )
    texts = text_splitter.split_text(clean_text)

    if progress_queue:
        progress_queue.put({"stage": "chunking_complete", "progress": 100, "percent": 25})

    # Summarize chunks using parallel processing
    chunk_summaries = summarize_chunks_parallel(texts, progress_queue)

    # Generate final summary
    final_summary = generate_final_summary(chunk_summaries, progress_queue)

    return {
        "summary": final_summary,
        "preprocessing_stats": stats,
        "total_chunks": len(texts)
    }

# ---------------- Flask Routes ----------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Global storage for processing tasks
processing_tasks = {}

@app.route('/upload', methods=['POST'])
def upload_file():
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

        # Create a unique task ID
        task_id = str(time.time())
        progress_queue = queue.Queue()
        result_queue = queue.Queue()

        # Store task info
        processing_tasks[task_id] = {
            "progress_queue": progress_queue,
            "result_queue": result_queue,
            "filepath": filepath
        }

        # Process in background thread
        def process_in_background():
            try:
                result = process_policy(filepath, progress_queue)
                result_queue.put({"success": True, "data": result})
            except Exception as e:
                result_queue.put({"success": False, "error": str(e)})
            finally:
                # Clean up uploaded file after processing
                if os.path.exists(filepath):
                    os.remove(filepath)

        thread = Thread(target=process_in_background)
        thread.start()

        return jsonify({"success": True, "task_id": task_id}), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    def generate():
        if task_id not in processing_tasks:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return

        task = processing_tasks[task_id]
        progress_queue = task["progress_queue"]
        result_queue = task["result_queue"]

        while True:
            # Check if processing is complete
            if not result_queue.empty():
                result = result_queue.get()
                yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"
                # Clean up task
                del processing_tasks[task_id]
                break

            # Check for progress updates
            if not progress_queue.empty():
                progress = progress_queue.get()
                yield f"data: {json.dumps({'type': 'progress', 'data': progress})}\n\n"

            time.sleep(0.1)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()

        if not data or 'pdf_path' not in data:
            return jsonify({"error": "pdf_path is required"}), 400

        pdf_path = data['pdf_path']

        if not os.path.exists(pdf_path):
            return jsonify({"error": f"PDF file not found: {pdf_path}"}), 404

        result = process_policy(pdf_path)

        return jsonify({
            "success": True,
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
