import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------- CONFIG ----------------
REFERENCE_PDF_FOLDER = "statement_6_insurance_documents\\vector_db"
CHROMA_DB_PATH = "chroma_store"
COLLECTION_NAME = "insurance_reference_docs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
    create_reference_db()
