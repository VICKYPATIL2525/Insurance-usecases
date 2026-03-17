# Import necessary libraries for file operations, PDF loading, embeddings, and vector storage
import os
from langchain_community.document_loaders import PyPDFLoader  # For loading and extracting text from PDF files
from langchain_huggingface import HuggingFaceEmbeddings      # For converting text to embeddings using HuggingFace models
from langchain_community.vectorstores import Chroma           # For storing and searching document embeddings


# ---------------- CONFIG ----------------
# Path to the folder containing PDF files to classify
TEST_PDF_FOLDER = "data\\testing"

# Directory where the Chroma vector database is stored
CHROMA_DB_PATH = "../shared/chroma_store"

# Name of the collection in the vector database (contains reference documents)
COLLECTION_NAME = "insurance_reference_docs"

# HuggingFace embedding model to convert text into numerical vectors
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- LOAD PDF ----------------
def load_pdf_text(pdf_path):
    """
    Loads a PDF file and extracts all text from it.

    Args:
        pdf_path: Full path to the PDF file

    Returns:
        String containing all text content from the PDF
    """
    # Initialize the PDF loader with the file path
    loader = PyPDFLoader(pdf_path)

    # Load the PDF and split into document objects (one per page)
    docs = loader.load()

    # Combine all pages' content into a single string separated by newlines
    return "\n".join(doc.page_content for doc in docs)


# ---------------- CLASSIFY ----------------
# This function takes a folder path, loads each PDF file, extracts its text, and classifies it by comparing against reference documents in a vector database using embedding similarity. It prints the classification results for each PDF.
def classify_folder(folder_path):
    """
    Classifies all PDF files in a folder by comparing them against reference documents
    stored in a vector database using embedding similarity.

    Args:
        folder_path: Path to the folder containing PDFs to classify
    """
    # Initialize the embedding model
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Connect to the existing Chroma vector database that contains reference documents
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,    # Where the database is stored
        collection_name=COLLECTION_NAME,      # Which collection to use
        embedding_function=embedding          # How to convert text to vectors
    )

    print("\n📂 Classifying documents...\n")

    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        # Skip non-PDF files
        if not file_name.lower().endswith(".pdf"):
            continue

        # Construct the full path to the PDF file
        pdf_path = os.path.join(folder_path, file_name)

        # Extract all text from the PDF to use as the query
        query_text = load_pdf_text(pdf_path)

        # Search the vector database for the top 3 most similar reference documents
        # Returns a list of (document, distance_score) tuples
        results = vectordb.similarity_search_with_score(
            query_text,  # The text to search for
            k=3          # Number of top matches to retrieve
        )

        # Initialize variables to track the best matching reference document
        best_match = None
        best_score = 0

        # Iterate through the search results to find the best match
        for doc, score in results:
            # Convert distance score to similarity percentage
            # Lower distance = higher similarity, so we do (1 - distance) * 100
            similarity = (1 - score) * 100

            # Extract the source file path from document metadata
            source = doc.metadata.get("source", "unknown")

            # Keep track of the reference document with the highest similarity
            if similarity > best_score:
                best_score = similarity
                best_match = source

        # ---- FINAL OUTPUT (NO LLM LOGIC NOW) ----
        # Display classification results for this PDF
        print(f"   📄 File: {file_name}")
        print(f"   ✅ Classified as: {best_match}")           # The most similar reference document
        print(f"   📊 Confidence: {best_score:.2f}%")         # How confident the classification is
        print("    🔧 Method: Embedding Similarity\n")         # Classification method used


# ---------------- MAIN ----------------
# Entry point of the script - runs when the file is executed directly
if __name__ == "__main__":
    # Start the classification process for all PDFs in the configured test folder
    classify_folder(TEST_PDF_FOLDER)
