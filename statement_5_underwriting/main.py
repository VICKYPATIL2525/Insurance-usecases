"""5. Underwriting Assistant
Description:
A GenAI-powered co-pilot that helps underwriters assess risk by summarizing applicant data, prior claims, and external reports.
Skills: Data summarization, LLM prompt chaining, structured output.
Demo idea: Feed sample applicant data → get risk score summary.
"""



import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# Load environment variables from .env file
load_dotenv()

# Folder Path
folder_path = r"data\Kamala_Venkatesh_HIGH"



class UnderwritingAnalysis(BaseModel):
    """Structured output model for underwriting analysis."""
    risk_score: int = Field(..., ge=0, le=100, description="Risk score between 0 and 100")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    risk_summary: str = Field(..., description="2-3 sentence overview of the risk")
    key_risk_factors: List[str] = Field(..., description="List of key risk factors")
    positive_indicators: List[str] = Field(..., description="List of positive indicators")
    underwriter_notes: str = Field(..., description="Short actionable note for underwriter")

llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.0,   # important for structured output
    max_tokens=1000,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30,  # Add timeout to prevent hanging
    max_retries=2,  # Reduce retries for faster failure
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


def read_all_pdfs_from_folder(folder_path: str, verbose: bool = True) -> dict:
    """
    Read all PDF files from a folder and return their contents.

    Args:
        folder_path: Path to the folder containing PDF files
        verbose: Whether to print detailed progress (default: True)

    Returns:
        Dictionary with filename as key and extracted text as value
    """
    results = {}

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return results

    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return results

    if verbose:
        print(f"Found {len(pdf_files)} PDF file(s) in '{folder_path}'")
        print("=" * 80)

    # Process each PDF
    for idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(folder_path, pdf_file)

        if verbose:
            print(f"\nProcessing [{idx}/{len(pdf_files)}]: {pdf_file}")
            print("-" * 40)
        else:
            print(f"Processing {pdf_file}...")

        try:
            # Extract text
            raw_text = extract_text_from_pdf(pdf_path)
            clean_text = preprocess_text(raw_text)

            results[pdf_file] = clean_text

            # Print preview only in verbose mode
            if verbose:
                print(f"Extracted {len(clean_text)} characters")
                print("\n--- Content Preview ---")
                print(clean_text[:300] + "..." if len(clean_text) > 300 else clean_text)
                print("-" * 40)
            else:
                print(f"  [OK] Extracted")

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            results[pdf_file] = None

    return results

def generate_underwriting_summary(documents: dict, max_chars_per_doc: int = 5000) -> UnderwritingAnalysis:
    """
    Takes extracted text from multiple underwriting documents
    and generates a final risk summary using LLM with structured output.

    Args:
        documents: Dictionary with filename as key and extracted text as value
        max_chars_per_doc: Maximum characters to use per document (default: 5000)

    Returns:
        UnderwritingAnalysis object with structured fields
    """

    # Combine documents with clear labels (with truncation for speed)
    combined_text = ""

    for filename, content in documents.items():
        if content:
            # Truncate very long documents to reduce API processing time
            truncated_content = content[:max_chars_per_doc]
            if len(content) > max_chars_per_doc:
                truncated_content += f"\n[... truncated {len(content) - max_chars_per_doc} characters ...]"
            combined_text += f"\n\n--- {filename} ---\n{truncated_content}"

    if not combined_text.strip():
        # Return a default structured response
        return UnderwritingAnalysis(
            risk_score=0,
            risk_level="UNKNOWN",
            risk_summary="No valid document content available for underwriting.",
            key_risk_factors=["No documents provided"],
            positive_indicators=[],
            underwriter_notes="Cannot proceed without documents."
        )

    # Create structured LLM with Pydantic output
    structured_llm = llm.with_structured_output(UnderwritingAnalysis)

    print(f"\nAnalyzing {len(documents)} documents...")

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


def format_underwriting_output(analysis: UnderwritingAnalysis) -> str:
    """Format the structured analysis into a readable string."""
    output = []
    output.append(f"Risk Score: {analysis.risk_score}")
    output.append(f"Risk Level: {analysis.risk_level}")
    output.append("")
    output.append("Risk Summary:")
    output.append(analysis.risk_summary)
    output.append("")

    # Only show Key Risk Factors if there are any
    if analysis.key_risk_factors:
        output.append("Key Risk Factors:")
        for factor in analysis.key_risk_factors:
            output.append(f"- {factor}")
        output.append("")

    # Only show Positive Indicators if there are any
    if analysis.positive_indicators:
        output.append("Positive Indicators:")
        for indicator in analysis.positive_indicators:
            output.append(f"- {indicator}")
        output.append("")

    output.append("Underwriter Notes:")
    output.append(analysis.underwriter_notes)

    return "\n".join(output)


if __name__ == "__main__":
    import time

    # Specify the folder path here
    folder_path = folder_path

    start_time = time.time()

    # Read all PDFs from the folder (dynamically handles any number of PDFs)
    # Set verbose=False for faster execution with less output
    print("Starting PDF extraction...")
    all_documents = read_all_pdfs_from_folder(folder_path, verbose=False)

    extraction_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total documents processed: {len(all_documents)}")
    print(f"Extraction time: {extraction_time:.2f} seconds")

    print("\n" + "=" * 80)
    print("UNDERWRITING CO-PILOT OUTPUT")
    print("=" * 80)

    # Generate structured analysis
    analysis_start = time.time()
    final_summary = generate_underwriting_summary(all_documents)
    analysis_time = time.time() - analysis_start

    # Format and print the output
    formatted_output = format_underwriting_output(final_summary)
    print(formatted_output)

    print("\n" + "=" * 80)
    print(f"Analysis time: {analysis_time:.2f} seconds")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("=" * 80)

