from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os
import uuid
import hashlib

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def ingest_pdf(file_path: str, vectordb_path: str):
    print(f"📥 Ingesting: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    docs = splitter.split_documents(pages)

    fhash = file_hash(file_path)

    # Add metadata
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source": os.path.basename(file_path),
            "page": doc.metadata.get("page", i),
            "layer": "company_policy",
            "priority": 1,
            "doc_id": str(uuid.uuid4()),
            "file_hash": fhash
        })

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        persist_directory=vectordb_path,
        embedding_function=embeddings
    )

    # 🚨 Prevent duplicate ingestion
    existing = db.get(where={"file_hash": fhash})
    if existing and len(existing.get("ids", [])) > 0:
        print("⚠️ File already exists. Skipping ingestion.")
        return

    db.add_documents(docs)
    db.persist()

    print(f"✅ Ingested {len(docs)} chunks")