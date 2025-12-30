import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- Paths ----------------
JSON_PATH = "st4data.json"
CHROMA_DIR = "chroma_store"

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
    persist_directory=CHROMA_DIR
)

print("💾 Persisting vector database to disk...")
vector_db.persist()

print("\n🎉 SUCCESS!")
print("✅ Quotes successfully stored in ChromaDB using MiniLM-L6-v2")
print(f"📁 Vector store location: {CHROMA_DIR}")
