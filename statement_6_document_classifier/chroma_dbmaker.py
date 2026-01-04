import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------- CONFIG ----------------
REFERENCE_PDF_FOLDER = "data\\vector_db"
CHROMA_DB_PATH = "../shared/chroma_store"
COLLECTION_NAME = "insurance_reference_docs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- Safety Check ----------------
def check_existing_collection():
    """Check if collection already exists and prompt user"""
    if not os.path.exists(CHROMA_DB_PATH):
        return True

    try:
        test_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        existing_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=test_embeddings
        )

        try:
            # Check collection count instead of similarity search (more reliable)
            count = existing_db._collection.count()

            if count > 0:
                print(f"⚠️  WARNING: Collection '{COLLECTION_NAME}' already exists with {count} documents!")
                print(f"📁 Location: {CHROMA_DB_PATH}")
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
            else:
                # Collection exists but is empty, delete and recreate
                print(f"Found empty collection '{COLLECTION_NAME}', recreating...\n")
                existing_db.delete_collection()
                return True

        except Exception as e:
            # Collection doesn't exist or error reading, proceed
            print(f"Collection check failed (will create new): {e}\n")
            return True

    except Exception as e:
        return True


# ---------------- LOAD PDFs ----------------
def load_reference_pdfs(folder_path):
    texts = []
    metadatas = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file_name))
            docs = loader.load()

            full_text = "\n".join(doc.page_content for doc in docs)

            texts.append(full_text)
            metadatas.append({
                "source": file_name
            })

    return texts, metadatas


# ---------------- CREATE DB ----------------
def create_reference_db():
    texts, metadatas = load_reference_pdfs(REFERENCE_PDF_FOLDER)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding
    )

    vectordb.add_texts(
        texts=texts,
        metadatas=metadatas
    )

    vectordb.persist()

    print("✅ ChromaDB populated successfully")
    print(f"📁 Collection: {COLLECTION_NAME}")
    print(f"📄 Documents added: {len(texts)}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    if check_existing_collection():
        create_reference_db()
