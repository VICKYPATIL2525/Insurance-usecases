"""2. Claims Description Normalizer
Description:
Develop an AI model that converts raw claim notes (free-text from adjusters or customers) into structured data—detecting loss type, severity, and affected asset.
Skills: NLP, entity extraction, prompt-based data structuring.
Demo idea: Input messy claim text → output clean JSON with claim attributes."""

# Import required libraries
import os
import json
import csv
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from typing import TypedDict, Optional
from dotenv import load_dotenv

# Schema definition for structured claim data extraction
class ClaimSchema(TypedDict):
    loss_type: Optional[str]        # Type of loss: Accident, Theft, Fire, Water Damage, Health, etc.
    severity: Optional[str]          # Severity level: Low, Medium, High
    affected_asset: Optional[str]    # Asset affected: Vehicle, Property, Health, Electronics, etc.


# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI LLM with specific configuration
llm = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini",
    temperature=0.0,   # important for structured output
    max_tokens=200,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Wrap LLM to enforce structured output according to ClaimSchema
structured_llm = llm.with_structured_output(ClaimSchema)

# Function to extract structured information from unstructured claim text
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
    # Invoke LLM with the prompt and return structured output
    return structured_llm.invoke(prompt)


def process_single_claim():
    """Process a single hardcoded claim"""
    print("\n=== SINGLE CLAIM PROCESSING ===\n")

    # Example claim text to process
    claim_text = """
Was driving home last night, heavy rain, car skidded and hit the divider.
Front bumper damaged badly, airbags deployed. Driver safe.
"""

    print(f"Processing claim...")
    # Extract structured data from the claim text
    result = normalize_claim_text(claim_text)
    print(f"\nExtracted Data:")
    print(result)

    # Save to JSON file with timestamp
    os.makedirs("jsonoutput", exist_ok=True)  # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")  # Format: DD-MM-YYYY_HH-MM-SS_AM/PM
    filename = f"jsonoutput/claim_output_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)  # Save with pretty formatting

    print(f"\nSaved to: {filename}")


def process_batch_claims():
    """Process multiple claims from CSV file"""
    print("\n=== BATCH CLAIM PROCESSING ===\n")

    csv_file = "statement_2_claims.csv"

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please ensure statement_2_claims.csv exists in the same directory.")
        return

    # Read CSV file (expected columns: claim_id, claim_text)
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        claims = list(reader)

    print(f"Found {len(claims)} claims to process\n")

    results = []  # Store all processed results
    os.makedirs("jsonoutput", exist_ok=True)  # Create output directory if it doesn't exist

    # Process each claim from the CSV file
    for idx, claim in enumerate(claims, 1):
        claim_id = claim['claim_id']  # Extract claim ID from CSV row
        claim_text = claim['claim_text']  # Extract claim description from CSV row

        print(f"Processing {idx}/{len(claims)}: {claim_id}...")

        try:
            # Extract structured data from claim text
            result = normalize_claim_text(claim_text)

            # Add claim_id and original claim_text to the result
            result_with_id = {
                "claim_id": claim_id,
                "claim_text": claim_text,
                **result  # Unpack the extracted fields (loss_type, severity, affected_asset)
            }

            results.append(result_with_id)  # Add to overall results list

            # Save individual JSON file for this claim
            filename = f"jsonoutput/{claim_id}.json"
            with open(filename, "w") as f:
                json.dump(result_with_id, f, indent=2)

            print(f"  ✓ Saved to: {filename}")

        except Exception as e:
            # Handle any errors that occur during processing
            print(f"  ✗ Error processing {claim_id}: {e}")

    # Save combined results to a single file with all claims
    timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")  # Format: DD-MM-YYYY_HH-MM-SS_AM/PM
    combined_filename = f"jsonoutput/batch_results_{timestamp}.json"
    with open(combined_filename, "w") as f:
        json.dump(results, f, indent=2)  # Save all results as a JSON array

    # Print summary of batch processing
    print(f"\n✓ Batch processing complete!")
    print(f"✓ Processed {len(results)}/{len(claims)} claims successfully")
    print(f"✓ Combined results saved to: {combined_filename}")


def main():
    """Main function to choose processing mode"""
    # Display application header
    print("=" * 50)
    print("INSURANCE CLAIMS NORMALIZER")
    print("=" * 50)
    print("\nChoose processing mode:")
    print("1. Single Claim Processing")
    print("2. Batch Processing (from CSV)")
    print("3. Exit")

    # Get user input for processing mode
    choice = input("\nEnter your choice (1/2/3): ").strip()

    # Route to appropriate processing function based on user choice
    if choice == "1":
        process_single_claim()  # Process one hardcoded claim
    elif choice == "2":
        process_batch_claims()  # Process multiple claims from CSV
    elif choice == "3":
        print("\nExiting...")  # Exit the application
    else:
        print("\nInvalid choice! Please run again and select 1, 2, or 3.")


# Entry point of the script
if __name__ == "__main__":
    main()
