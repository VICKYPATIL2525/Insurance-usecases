'''1. Policy Summary Assistant
Description:
Build a GenAI tool that can read lengthy insurance policy documents and generate a concise, plain-language summary of coverage, exclusions, and limits.
Skills: Text summarization, NLP, LLM fine-tuning.
Demo idea: Upload a sample policy PDF → auto-generate a 200-word summary.'''

import os
import re
import time  # ⚡ NEW: Added for performance tracking
from dotenv import load_dotenv

# ---------------- Load env ----------------
load_dotenv()

# ---------------- PDF Loading ----------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage, SystemMessage

# ---------------- Path ----------------
path = r"data\health_insurance_document.pdf"


# ---------------- Extract PDF Text ----------------
# ✅ UNCHANGED: Same as original
def extract_text_from_pdf_langchain(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)


pdf_text = extract_text_from_pdf_langchain(path)

print("===== RAW EXTRACTED TEXT =====")
print(pdf_text[:1000])
print("=" * 150)


# ---------------- Simple Lossless Preprocessing ----------------
# ✅ UNCHANGED: Same as original
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


clean_text, stats = preprocess_text_basic(pdf_text)

print("---- Preprocessing Stats ----")
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n---- Cleaned Text Preview ----")
print(clean_text[:1000])


# ---------------- Azure OpenAI LLM ----------------
# ✅ UNCHANGED: Same as original
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.1,
    max_tokens=400,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ---------------- Chunking ----------------
# ✅ UNCHANGED: Same as original
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=150
)

texts = text_splitter.split_text(clean_text)
print(f"\nTotal chunks created: {len(texts)}")


# ===============================================================================
# ⚡ OPTIMIZATION START: PARALLEL BATCH PROCESSING
# ===============================================================================
# 🔥 KEY CHANGE: Using llm.batch() instead of sequential for-loop
# This processes multiple chunks in parallel, drastically reducing time
# ===============================================================================

# ---------------- MAP: Summarize Chunks (OPTIMIZED) ----------------
def summarize_chunks_parallel(chunks: list[str]) -> list[str]:
    """
    ⚡ OPTIMIZED VERSION using LangChain's batch() method

    WHAT CHANGED:
    - Original: Sequential for-loop calling llm.invoke() one by one
    - New: Single llm.batch() call that processes all chunks in parallel

    PERFORMANCE IMPACT:
    - Original: 10 chunks × 5 seconds = 50 seconds
    - New: 10 chunks with max_concurrency=5 ≈ 10-15 seconds
    - Speedup: ~3-5x faster
    """

    # Same system prompt as original
    system_prompt = (
        "You are an insurance domain expert. "
        "Summarize the following section of a policy document. "
        "Extract facts only. Do NOT add assumptions. "
        "Focus on coverage, exclusions, limits, waiting periods, and conditions."
    )

    # ⚡ NEW: Build all messages upfront (instead of inside loop)
    # Create a list of message lists - one for each chunk
    all_messages = [
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=chunk)
        ]
        for chunk in chunks
    ]

    print(f"\n⚡ Processing {len(chunks)} chunks in parallel...")
    start_time = time.time()

    # ⚡ NEW: Single batch call instead of for-loop
    # max_concurrency controls how many API calls run simultaneously
    # Adjust this based on your Azure OpenAI rate limits (5-10 is safe)
    responses = llm.batch(
        all_messages,
        config={"max_concurrency": 5}  # Process 5 chunks at a time
    )

    # ⚡ NEW: Extract content from all responses
    summaries = [response.content for response in responses]

    elapsed_time = time.time() - start_time
    print(f"✅ All {len(chunks)} chunks summarized in {elapsed_time:.2f} seconds")
    print(f"⚡ Average time per chunk: {elapsed_time/len(chunks):.2f} seconds")

    return summaries


# ⚡ NEW: Call the optimized parallel version
chunk_summaries = summarize_chunks_parallel(texts)

# ===============================================================================
# ⚡ OPTIMIZATION END
# ===============================================================================


# ---------------- REDUCE: Final Summary ----------------
# ✅ UNCHANGED: Same as original (no optimization needed - single call)
def generate_final_summary(chunk_summaries: list[str]) -> str:
    combined_summary = "\n".join(chunk_summaries)

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

    response = llm.invoke(messages)
    return response.content


final_summary = generate_final_summary(chunk_summaries)

print("\n===== FINAL POLICY SUMMARY =====\n")
print(final_summary)


# ===============================================================================
# 📊 PERFORMANCE COMPARISON SUMMARY
# ===============================================================================
#
# ORIGINAL CODE (statement_1.py):
# - Method: Sequential for-loop with llm.invoke()
# - Time for 10 chunks: ~50 seconds
# - Bottleneck: Waits for each API call to complete before starting next
#
# OPTIMIZED CODE (this file):
# - Method: Parallel batch processing with llm.batch()
# - Time for 10 chunks: ~10-15 seconds
# - Speedup: 3-5x faster
# - Key parameter: max_concurrency=5 (adjust based on rate limits)
#
# WHAT TO ADJUST:
# 1. max_concurrency: Increase to 10 if you have higher rate limits
# 2. temperature: Can increase to 0.3 for slightly faster generation
# 3. max_tokens: Reduce from 400 to 300 if summaries are shorter
#
# ===============================================================================
