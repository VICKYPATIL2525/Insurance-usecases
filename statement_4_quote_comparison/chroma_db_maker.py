import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- Paths ----------------
JSON_PATH = "data/statement_4_data.json"
CHROMA_DIR = "../shared/chroma_store"
COLLECTION_NAME = "insurance_quotes"


# ---------------- Safety Check ----------------
def check_existing_collection():
    """Check if collection already exists and prompt user"""
    if not os.path.exists(CHROMA_DIR):
        return True  # First time, proceed

    try:
        # Try to load existing collection
        test_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        existing_db = Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=test_embeddings
        )

        # Check if collection has data
        try:
            test_results = existing_db.similarity_search("test", k=1)
            if test_results:
                print(f"⚠️  WARNING: Collection '{COLLECTION_NAME}' already exists with data!")
                print(f"📁 Location: {CHROMA_DIR}")
                print("\nOptions:")
                print("  1. SKIP - Keep existing data and exit (recommended)")
                print("  2. RECREATE - Delete and rebuild collection")

                choice = input("\nEnter choice (1 or 2): ").strip()

                if choice == "2":
                    print("\n🗑️  Deleting existing collection...")
                    existing_db.delete_collection()
                    print("✅ Collection deleted. Proceeding with creation...\n")
                    return True
                else:
                    print("\n✅ Keeping existing collection. Exiting...")
                    return False
        except:
            # Collection exists but is empty, proceed
            return True

    except Exception as e:
        # Collection doesn't exist or error reading, proceed
        return True


# Check before proceeding
if not check_existing_collection():
    exit(0)

print("🔹 Starting insurance quote ingestion pipeline...\n")

# ---------------- Load JSON ----------------
print(f"📂 Loading data from: {JSON_PATH}")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded raw JSON with {len(data)} entries\n")

# ---------------- Convert to LangChain Documents ----------------
print("🧩 Converting JSON entries into LangChain Documents...")
documents = []

for idx, item in enumerate(data, start=1):
    doc = Document(
        page_content=item["content"],   # text for embeddings
        metadata=item["metadata"]        # structured metadata
    )
    documents.append(doc)

    # Lightweight progress update
    print(f"   ✔ Prepared document {idx}/{len(data)}")

print("\n✅ All documents prepared successfully\n")

# ---------------- MiniLM Embeddings ----------------
print("🧠 Initializing MiniLM-L6-v2 embedding model (local, free)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embedding model loaded\n")

# ---------------- Create Chroma Vector DB ----------------
print("📦 Creating Chroma vector database...")
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
)

print("💾 Persisting vector database to disk...")
vector_db.persist()

print("\n🎉 SUCCESS!")
print("✅ Quotes successfully stored in ChromaDB using MiniLM-L6-v2")
print(f"📁 Vector store location: {CHROMA_DIR}")
print(f"📦 Collection name: {COLLECTION_NAME}")
