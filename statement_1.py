'''1. Policy Summary Assistant
Description:
Build a GenAI tool that can read lengthy insurance policy documents and generate a concise, plain-language summary of
coverage, exclusions, and limits.
Skills: Text summarization, NLP, LLM fine-tuning.
Demo idea: Upload a sample policy PDF → auto-generate a 200-word summary.'''


import os
import re
from dotenv import load_dotenv

# ---------------- Load env ----------------
load_dotenv()

# ---------------- PDF Loading ----------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.messages import HumanMessage, SystemMessage

# ---------------- Path ----------------
path = r"uploads\health_insurance_document.pdf"   # fixed escape issue


# ---------------- Extract PDF Text ----------------
def extract_text_from_pdf_langchain(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)


pdf_text = extract_text_from_pdf_langchain(path)

print("===== RAW EXTRACTED TEXT =====")
print(pdf_text[:1000])
print("=" * 150)


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


clean_text, stats = preprocess_text_basic(pdf_text)

print("---- Preprocessing Stats ----")
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n---- Cleaned Text Preview ----")
print(clean_text[:1000])


# ---------------- Azure OpenAI LLM ----------------
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.1,
    max_tokens=400,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ---------------- Chunking ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=150
)

texts = text_splitter.split_text(clean_text)
print(f"\nTotal chunks created: {len(texts)}")


# ---------------- MAP: Summarize Chunks ----------------
def summarize_chunks(chunks: list[str]) -> list[str]:
    summaries = []

    system_prompt = (
        "You are an insurance domain expert. "
        "Summarize the following section of a policy document. "
        "Extract facts only. Do NOT add assumptions. "
        "Focus on coverage, exclusions, limits, waiting periods, and conditions."
    )

    for idx, chunk in enumerate(chunks, start=1):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=chunk)
        ]

        response = llm.invoke(messages)
        summaries.append(response.content)

        print(f"✔ Chunk {idx}/{len(chunks)} summarized")
    
    return summaries


chunk_summaries = summarize_chunks(texts)


# ---------------- REDUCE: Final Summary ----------------
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
