#!/usr/bin/env python3
"""
rag_builder.py — Production-Grade RAG Vector DB Builder v5
===========================================================
FIXES vs v4:
  FIX-1  Embedding model unified → BAAI/bge-base-en-v1.5 (matches agentic_rag.py)
  FIX-2  Collection name unified → "agentic_rag_collection" (was missing in RAG → read empty DB)
  FIX-3  Chunk size 450→800, overlap 100→150 → fewer mid-sentence cuts, richer context
  FIX-4  SemanticChunker (sentence-aware) replaces dumb RecursiveCharacterTextSplitter
         Falls back to RecursiveCharacterTextSplitter if sentence-transformers unavailable
  FIX-5  Minimum content length filter → skip chunks < 60 chars (noise / page headers)
  FIX-6  Page number stored as int (not string) for reliable metadata filtering
  FIX-7  de-duplication by content hash before insert → no repeated chunks in DB
  FIX-8  Verify collection count after build and warn if 0

Usage:
    python rag_builder.py               # incremental (only changed files)
    python rag_builder.py --force       # re-index ALL files
    python rag_builder.py --debug       # verbose output
    python rag_builder.py --pdf-dir /your/path/to/pdfs
"""

import os
import sys
import re
import json
import hashlib
import time
import argparse
from typing import Dict, List, Tuple

import torch


def clean_pdf_text(text: str) -> str:
    """Cleans up vertical newline spacing issues in PDF text, collapsing space artifacts
    while preserving bulleted/numbered lists and section boundaries."""
    # 1. Collapse all consecutive whitespace to a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Restore newlines before structural list and section markers
    # Numbered sections: e.g. " 10.4 - " or " 10.4.1 - " -> "\n10.4 - "
    text = re.sub(r'\s+(\d+(\.\d+)+\s*-)', r'\n\1', text)
    text = re.sub(r'\s+(\d+\.\w+\s*-)', r'\n\1', text)  # also match things like 10.11.C
    text = re.sub(r'\s+(\d+\.\d+\s*-)', r'\n\1', text)
    
    # Capital letter sections: e.g. " A. " or " B. Excluded" -> "\n\nA. " or "\n\nB. "
    text = re.sub(r'\s+([A-Z]\.\s+[A-Z])', r'\n\n\1', text)
    
    # Numbered list items in parentheses or dot format: e.g. " (1) " -> "\n(1) ", " 1. " -> "\n1. "
    text = re.sub(r'\s+(\(\d+\)\s+)', r'\n\1', text)
    text = re.sub(r'\s+(\d+\.\s+[A-Z])', r'\n\1', text)
    
    # Lettered lists: e.g. " (a) " or " a) " -> "\n(a) ", "\na) "
    text = re.sub(r'\s+(\([a-z]\)\s+)', r'\n\1', text)
    text = re.sub(r'\s+([a-z]\)\s+)', r'\n\1', text)
    
    # Bullet points: e.g. " • " -> "\n• "
    text = re.sub(r'\s+(•\s+)', r'\n\1', text)
    
    return text.strip()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — must match agentic_rag.py exactly
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT   = "/home/ubuntu/projects/RAG" if os.path.exists("/home/ubuntu/projects/RAG") else os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR  = os.path.join(PROJECT_ROOT, "chroma_db_fast")
HASH_FILE      = os.path.join(PROJECT_ROOT, "doc_hashes.json")

# Fallback PDF_DIR: check if pdf/ exists, otherwise use project root itself
PDF_DIR        = os.path.join(PROJECT_ROOT, "pdf")
if not os.path.exists(PDF_DIR) or not any(f.lower().endswith(".pdf") for f in os.listdir(PDF_DIR)):
    PDF_DIR    = PROJECT_ROOT

# FIX-1: MUST match the embedding model used in agentic_rag.py
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# FIX-2: MUST match the collection name used in agentic_rag.py
COLLECTION_NAME = "agentic_rag_collection"

LAYER_PRIORITY = {
    "company_policy": 1,
    "medicare":       2,
    "medical":        3,
    "general":        4,
}

# FIX-3: Larger chunks → more context per retrieval, less mid-sentence truncation
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LEN = 60    # FIX-5: skip tiny noise chunks
BATCH_SIZE    = 32    # smaller batches → more stable memory usage

# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CHUNKER (FIX-4)
# ══════════════════════════════════════════════════════════════════════════════

def make_splitter():
    """
    FIX-4: Use sentence-aware splitting.
    Tries langchain_experimental SemanticChunker first (groups semantically
    similar sentences). Falls back to recursive splitter if not installed.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings as _Emb
        emb = _Emb(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  [Chunker] Using SemanticChunker (sentence-aware)")
        return SemanticChunker(emb, breakpoint_threshold_type="percentile")
    except ImportError:
        pass

    # Fallback: still use sentence-boundary separators
    print("  [Chunker] SemanticChunker not available -> RecursiveCharacterTextSplitter "
          "(sentence-aware separators)")
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_pdf_files(root: str) -> List[Tuple[str, str]]:
    results = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                abs_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(abs_path, root)
                results.append((abs_path, rel_path))
    results.sort()
    return results


def detect_layer(rel_path: str, docs=None) -> Tuple[str, int]:
    path_lower = rel_path.replace("\\", "/").lower()
    parts = path_lower.split("/")
    
    # 1. Directory based detection
    if len(parts) > 1:
        first = parts[0]
        if first in LAYER_PRIORITY:
            return first, LAYER_PRIORITY[first]
            
    # 2. Filename keyword based detection for flat files
    filename = parts[-1]
    if any(x in filename for x in ("handbook", "pto", "employee", "abbreviation")):
        return "company_policy", 1
    elif "medicare" in filename:
        return "medicare", 2
    elif "medical" in filename:
        return "medical", 3
    elif "policy" in filename:
        # Check content sample for generic "policy" filename
        if docs:
            text_sample = " ".join([d.page_content.lower()[:300] for d in docs[:3]])
            if any(w in text_sample for w in ("medicare", "hha", "noa", "pps")):
                return "medicare", 2
        return "company_policy", 1
        
    # 3. Content keyword based detection if flat file has generic name
    if docs:
        text_sample = " ".join([d.page_content.lower()[:800] for d in docs[:3]])
        # Medicare signals
        medicare_signals = ["medicare", "hha", "noa", "pps", "lupa", "notice of admission", "home health claims"]
        if any(sig in text_sample for sig in medicare_signals):
            return "medicare", 2
        # Company policy signals
        policy_signals = ["pto", "employee handbook", "elumina", "paid time off", "abbreviations", "benefits", "vacation"]
        if any(sig in text_sample for sig in policy_signals):
            return "company_policy", 1

    return "general", 4


def load_hashes() -> Dict[str, str]:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE) as f:
            return json.load(f)
    return {}


def save_hashes(h: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(HASH_FILE) or ".", exist_ok=True)
    with open(HASH_FILE, "w") as f:
        json.dump(h, f, indent=2)


def make_chunk_id(rel_path: str, page: int, chunk_idx: int) -> str:
    base = f"{rel_path}#p{page}#c{chunk_idx}"
    return hashlib.md5(base.encode()).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.md5(text[:300].encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# CORE INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def ingest_single_pdf(db: Chroma, abs_path: str, rel_path: str,
                      splitter, debug: bool = False) -> int:
    # Load PDF first to inspect content for layer classification if filename keywords are insufficient
    try:
        loader = PyPDFLoader(abs_path)
        docs   = loader.load()
        if not docs:
            print(f"    [WARN] PDF produced 0 pages — skipping")
            return 0
        if debug:
            print(f"    Loaded {len(docs)} pages")
    except Exception as e:
        print(f"    [ERROR] Could not load PDF: {e}")
        return 0

    layer, priority = detect_layer(rel_path, docs)
    print(f"\n  Processing: {rel_path}")
    print(f"    layer={layer}  priority={priority}")

    # Delete old chunks for this file (prevents duplicates on re-index)
    try:
        db.delete(where={"path": rel_path})
        if debug:
            print("    Deleted old entries")
    except Exception as e:
        if debug:
            print(f"    [WARN] Delete failed (OK on first run): {e}")

    # Enrich metadata — FIX-6: store page as int
    ts = time.time()
    for d in docs:
        d.page_content = clean_pdf_text(d.page_content)
        raw_page = d.metadata.get("page", 0)
        d.metadata.update({
            "source":    os.path.basename(rel_path),
            "path":      rel_path,
            "layer":     layer,
            "priority":  priority,
            "timestamp": ts,
            "page":      int(raw_page) if str(raw_page).isdigit() else 0,
        })

    # Page-Level Ingestion: keep each page as a single chunk
    chunks = docs

    if not chunks:
        print(f"    [WARN] No chunks produced — skipping")
        return 0

    # FIX-5: Filter noise chunks shorter than MIN_CHUNK_LEN
    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
    if debug and before != len(chunks):
        print(f"    Filtered {before - len(chunks)} noise chunks (< {MIN_CHUNK_LEN} chars)")

    # FIX-7: De-duplicate by content hash before insert
    seen_hashes: set = set()
    unique_chunks, unique_ids = [], []
    for i, c in enumerate(chunks):
        h = content_hash(c.page_content)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        unique_chunks.append(c)
        page = int(c.metadata.get("page", 0))
        unique_ids.append(make_chunk_id(rel_path, page, i))

    if debug:
        print(f"    {len(chunks)} chunks → {len(unique_chunks)} unique after dedup")

    # Batch insert
    for i in range(0, len(unique_chunks), BATCH_SIZE):
        db.add_documents(unique_chunks[i:i + BATCH_SIZE],
                         ids=unique_ids[i:i + BATCH_SIZE])

    print(f"    Added {len(unique_chunks)} chunks")
    return len(unique_chunks)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — called by agentic_rag.py on hot-reload
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdf(file_path: str, vector_db_dir: str) -> int:
    """
    Single-file ingestion API used by AgenticRAGOrchestrator.ingest_document().
    Returns number of chunks added.
    """
    device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        collection_name=COLLECTION_NAME,   # FIX-2
        persist_directory=vector_db_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    splitter  = make_splitter()
    rel_path  = os.path.basename(file_path)
    return ingest_single_pdf(db, file_path, rel_path, splitter, debug=False)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Vector DB Builder v5")
    parser.add_argument("--force",   action="store_true",
                        help="Re-index ALL files, ignoring cached hashes")
    parser.add_argument("--debug",   action="store_true",
                        help="Verbose output")
    parser.add_argument("--pdf-dir", default=PDF_DIR,
                        help=f"Override PDF directory (default: {PDF_DIR})")
    args    = parser.parse_args()
    pdf_dir = args.pdf_dir

    print("\n" + "=" * 60)
    print("RAG Builder v5 — starting")
    print("=" * 60)
    print(f"  PDF_DIR        : {pdf_dir}")
    print(f"  VECTOR_DB_DIR  : {VECTOR_DB_DIR}")
    print(f"  COLLECTION     : {COLLECTION_NAME}")
    print(f"  EMBEDDING MODEL: {EMBEDDING_MODEL}")
    print(f"  HASH_FILE      : {HASH_FILE}")
    print(f"  Force rebuild  : {args.force}")
    print("=" * 60)

    # Verify PDF_DIR
    if not os.path.exists(pdf_dir):
        print(f"\n[ERROR] PDF_DIR does not exist: {pdf_dir}")
        print(f"  mkdir -p {pdf_dir}/company_policy && cp your.pdf {pdf_dir}/company_policy/")
        sys.exit(1)

    all_pdfs = iter_pdf_files(pdf_dir)
    print(f"\n  Found {len(all_pdfs)} PDF(s) in {pdf_dir}")
    if args.debug:
        for _, rel in all_pdfs:
            print(f"    {rel}")

    if not all_pdfs:
        print("\n[ERROR] No PDF files found!")
        print(f"\n  Expected structure:")
        print(f"    {pdf_dir}/company_policy/handbook.pdf")
        print(f"    {pdf_dir}/medicare/plan.pdf")
        sys.exit(1)

    # Check existing DB
    db_is_empty = False
    chroma_db_file = os.path.join(VECTOR_DB_DIR, "chroma.sqlite3")
    if not os.path.exists(VECTOR_DB_DIR) or not os.path.exists(chroma_db_file):
        db_is_empty = True
        print("\n  Vector DB not found — will build from scratch")
    else:
        try:
            _emb = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs={"device": "cpu"},
            )
            _db  = Chroma(
                collection_name=COLLECTION_NAME,   # FIX-2
                persist_directory=VECTOR_DB_DIR,
                embedding_function=_emb,
                collection_metadata={"hnsw:space": "cosine"},
            )
            doc_count = _db._collection.count()
            if doc_count == 0:
                db_is_empty = True
                print(f"\n  Vector DB is EMPTY (0 docs) — will rebuild")
            else:
                print(f"\n  Vector DB has {doc_count} existing chunk(s)")
        except Exception as e:
            db_is_empty = True
            print(f"\n  Could not read vector DB ({e}) — will rebuild")

    # Load hashes
    if args.force or db_is_empty:
        print("\n  Re-indexing ALL files")
        prev_hashes = {}
        if os.path.exists(HASH_FILE):
            os.remove(HASH_FILE)
    else:
        prev_hashes = load_hashes()
        if args.debug:
            print(f"\n  Loaded {len(prev_hashes)} cached hash(es)")

    new_hashes: Dict[str, str] = {}
    to_process: List[Tuple[str, str]] = []

    for abs_path, rel_path in all_pdfs:
        h = sha256_file(abs_path)
        new_hashes[rel_path] = h
        if prev_hashes.get(rel_path) != h:
            to_process.append((abs_path, rel_path))
            if args.debug:
                reason = "new" if rel_path not in prev_hashes else "changed"
                print(f"    QUEUE ({reason}): {rel_path}")
        elif args.debug:
            print(f"    SKIP  (unchanged): {rel_path}")

    if not to_process:
        print("\n  No new or changed PDFs detected — DB is up-to-date.")
        print("  To force a full re-index: python rag_builder.py --force")
        sys.exit(0)

    print(f"\n  {len(to_process)} file(s) to index "
          f"({len(all_pdfs) - len(to_process)} unchanged/skipped)")

    # Setup embeddings + Chroma
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    device = "cpu"
    print(f"\n  Embedding device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        collection_name=COLLECTION_NAME,   # FIX-2
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    splitter = make_splitter()

    # Process
    print(f"\n  Indexing {len(to_process)} file(s) ...\n")
    total_chunks = 0
    failed = 0

    for abs_path, rel_path in to_process:
        try:
            n = ingest_single_pdf(db, abs_path, rel_path, splitter, debug=args.debug)
            total_chunks += n
        except Exception as e:
            print(f"    [ERROR] {rel_path}: {e}")
            failed += 1

    save_hashes(new_hashes)

    print("\n" + "=" * 60)
    print("Build complete")
    print(f"  Files indexed : {len(to_process) - failed}")
    print(f"  Files failed  : {failed}")
    print(f"  Total chunks  : {total_chunks}")
    print(f"  Vector DB     : {VECTOR_DB_DIR}")
    print(f"  Collection    : {COLLECTION_NAME}")
    print("=" * 60)

    # FIX-8: Verify final count
    try:
        count = db._collection.count()
        print(f"\n  ✅ Chroma collection '{COLLECTION_NAME}': {count} chunk(s) total")
        if count == 0:
            print("  ⚠️  WARNING: 0 chunks in DB — check PDF loading errors above!")
    except Exception:
        pass


if __name__ == "__main__":
    main()
