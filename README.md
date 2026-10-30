**BioRAG** is an agentic, hybrid search Retrieval Augmented Generation (RAG) system engineered for deep semantic analysis and question answering across peer reviewed computational biology, molecular chemistry, and structural bioinformatics literature.

The pipeline combines dense vector embeddings with sparse BM25 lexical indexing and cross encoder reranking to accurately resolve technical domain queries spanning $SE(3)$-equivariant geometric deep learning, CRISPR-Cas9 screening pipelines, and low data drug discovery frameworks.



## Architecture Overview
                  ┌─────────────────────────┐
                  │    Input Query (User)   │
                  └────────────┬────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
    ┌───────────────────────┐     ┌───────────────────────┐
    │ Dense Vector Search   │     │ Sparse Lexical Search │
    │ (BAAI/bge-base-en)    │     │ (BM25Okapi)           │
    └───────────┬───────────┘     └───────────┬───────────┘
                │                             │
                └──────────────┬──────────────┘
                               │ (Candidate Chunks)
                               ▼
                  ┌─────────────────────────┐
                  │  Cross-Encoder Reranker │
                  │  (ms-marco-MiniLM-L6)   │
                  └────────────┬────────────┘
                               │ (Top-k Reranked Context)
                               ▼
                  ┌─────────────────────────┐
                  │  LLM Synthesis Engine   │
                  │  (vLLM / Open Endpoint) │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Synthesized Answer +    │
                  │ Provenance Citations    │
                  └─────────────────────────┘



## Core Technical Features

* **Hybrid Two-Stage Retrieval:**
  * **Dense Semantic Search:** Uses `BAAI/bge-base-en-v1.5` embeddings (768-dimensional) to map semantic relationships across complex scientific text.
  * **Sparse Lexical Search:** Employs `BM25Okapi` to capture precise keyword terminology (e.g., exact gene names, chemical abbreviations, enzyme variants).
* **Cross-Encoder Reranking:** Integrates `cross-encoder/ms-marco-MiniLM-L-6-v2` to score and rerank candidate chunk pairs before context injection, significantly minimizing hallucinations.
* **Persistent Vector Store:** Leverages **ChromaDB** with sentence-aware recursive text chunking for vector persistence and metadata filtering.
* **Resilient Infrastructure:** Features an automated in-process memory fallback when external caching engines (e.g., Redis) are unavailable.
* **Dual Interface Support:** Includes a lightweight **Flask REST API** backend and an interactive **Gradio** web user interface.



## Domain Knowledge Base Focus

The default vector collection is indexed on peer-reviewed biomedical literature covering:

| Domain Focus | Key Concepts & Methodologies |
| :--- | :--- |
| **Geometric Deep Learning (GDL)** | $SE(3)$-equivariance, GVP-GNNs, E(n)-EGNNs, 3D point clouds, diffusion docking (DiffDock). |
| **Small-Molecule Drug Discovery** | Structure-based & ligand-based virtual screening, de novo design, QSAR/QSPR modeling, ADMET profiling. |
| **CRISPR-Cas9 Bioinformatics** | Guide RNA design, off-target prediction (Cas-OFFinder), essentiality screening (MAGeCK, CERES). |
| **Data-Limited Machine Learning** | Transfer learning, self-supervised representation learning (SSL), few-shot meta-learning in drug discovery. |

---

## Project Structure
BioRAG/
├── pdf/                          # Source domain PDF literature
├── chroma_db_fast/               # Persistent ChromaDB vector collections
├── cache/                        # Local Hugging Face model cache
├── doc_hashes.json               # SHA-256 document state & change tracking
├── ingestion.py                  # Document parsing and sentence-aware chunking
├── rag_builder.py                # Vector store & BM25 index builder
├── agentic_rag.py                # Core hybrid retrieval & reranking logic
├── agentic_app.py                # Flask API server & application entrypoint
├── gardio_ui.py                  # Interactive Gradio interface
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/BioRAG.git
cd BioRAG

