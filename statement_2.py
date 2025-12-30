"""2. Claims Description Normalizer
Description:
Develop an AI model that converts raw claim notes (free-text from adjusters or customers) into structured data—detecting
loss type, severity, and affected asset.
Skills: NLP, entity extraction, prompt-based data structuring.
Demo idea: Input messy claim text → output clean JSON with claim attributes."""


from typing import TypedDict, Optional
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Schema definition for structured claim data extraction
class ClaimSchema(TypedDict):
    loss_type: Optional[str]        # Type of loss: Accident, Theft, Fire, Water Damage, Health, etc.
    severity: Optional[str]          # Severity level: Low, Medium, High
    affected_asset: Optional[str]    # Asset affected: Vehicle, Property, Health, Electronics, etc.


load_dotenv()

# Example claim texts for testing:
#
# Example 1 - Vehicle Accident:
# "Was driving home last night, heavy rain, car skidded and hit the divider.
# Front bumper damaged badly, airbags deployed. Driver safe."
#
# Example 2 - Home Fire:
# "Kitchen caught fire while cooking. Extensive smoke damage to walls and ceiling.
# Microwave and cabinets destroyed. Family evacuated safely."
#
# Example 3 - Theft:
# "Returned from vacation to find back door broken. TV, laptop, and jewelry missing.
# Police report filed. Estimated loss around $5000."
#
# Example 4 - Water Damage:
# "Woke up to water leaking from ceiling. Pipe burst in the bathroom above.
# Bedroom carpet and furniture soaked. Need immediate repairs."
#
# Example 5 - Health/Medical:
# "Slipped on wet floor at grocery store. Fractured wrist, required emergency visit.
# X-rays done, cast applied. Following up with orthopedic specialist."
#
# Example 6 - Electronics:
# "Lightning strike during storm. Power surge fried my desktop computer,
# gaming console, and router. Smoke smell from PC."

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

# Example claim text to process
claim_text = """
Was driving home last night, heavy rain, car skidded and hit the divider.
Front bumper damaged badly, airbags deployed. Driver safe.
"""

# Process the claim text and extract structured data
result = normalize_claim_text(claim_text)
print(result)

# Save to JSON file with timestamp
os.makedirs("statement_2_json_output", exist_ok=True)  # Create output directory if it doesn't exist
timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")  # Format: DD-MM-YYYY_HH-MM-SS_AM/PM
filename = f"jsonoutput/claim_output_{timestamp}.json"
with open(filename, "w") as f:
    json.dump(result, f, indent=2)  # Save with pretty formatting