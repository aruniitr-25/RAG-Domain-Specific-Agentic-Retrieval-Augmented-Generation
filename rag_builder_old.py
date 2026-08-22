#!/usr/bin/env python3
"""
diagnose_db.py — Check what's actually in your Chroma vector DB
Run: python diagnose_db.py
"""
import os
import sys

# ── MATCH THESE TO YOUR agentic_rag.py CONFIG ────────────────────────────────
VECTOR_DB_DIR = "/home/ubuntu/projects/RAG/chroma_db_fast"

# ── Check ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Vector DB Diagnostic")
print("="*60)

print(f"\n1. Directory: {VECTOR_DB_DIR}")
if not os.path.exists(VECTOR_DB_DIR):
    print("   ❌ DOES NOT EXIST")
    sys.exit(1)
else:
    print("   ✅ Exists")
    files = os.listdir(VECTOR_DB_DIR)
    print(f"   Files: {files}")

# Check for SQLite DB
sqlite_path = os.path.join(VECTOR_DB_DIR, "chroma.sqlite3")
if os.path.exists(sqlite_path):
    size = os.path.getsize(sqlite_path)
    print(f"\n2. chroma.sqlite3: {size:,} bytes")
    if size < 50000:
        print("   ⚠️  Very small — likely empty or corrupt")
else:
    print("\n2. chroma.sqlite3: ❌ NOT FOUND — DB was never built")

# Try connecting and listing ALL collections
print("\n3. Connecting to Chroma...")
try:
    import chromadb
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collections = client.list_collections()
    print(f"   Collections found: {len(collections)}")
    for col in collections:
        count = col.count()
        print(f"   - '{col.name}': {count} documents")
        if count == 0:
            print("     ⚠️  EMPTY — this is why RAG finds 0 docs")
except Exception as e:
    print(f"   Error: {e}")

# Try with langchain
print("\n4. Trying LangChain Chroma (default collection)...")
try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    # Try default collection name
    db1 = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=emb)
    count1 = db1._collection.count()
    print(f"   Default collection: {count1} docs")

    # Try named collection (from builder)
    db2 = Chroma(
        collection_name="agentic_rag_collection",
        persist_directory=VECTOR_DB_DIR,
        embedding_function=emb,
    )
    count2 = db2._collection.count()
    print(f"   'agentic_rag_collection': {count2} docs")

    if count1 > 0:
        print("\n   ✅ Documents are in the DEFAULT collection")
        print("   FIX: Your agentic_rag.py must NOT use collection_name=")
        # Show sample
        sample = db1.get(limit=2)
        print(f"   Sample metadata: {sample['metadatas'][:2]}")
    elif count2 > 0:
        print("\n   ✅ Documents are in 'agentic_rag_collection'")
        print("   FIX: Your agentic_rag.py must use collection_name='agentic_rag_collection'")
        sample = db2.get(limit=2)
        print(f"   Sample metadata: {sample['metadatas'][:2]}")
    else:
        print("\n   ❌ BOTH collections are EMPTY")
        print("   FIX: Run the builder first:")
        print("     python rag_builder.py --force")

except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("ACTION NEEDED:")
print("  If both collections are empty → run: python rag_builder.py --force")
print("  If docs are in wrong collection → update VECTOR_DB_DIR in agentic_rag.py")
print("="*60 + "\n")