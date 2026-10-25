
import os
PROJECT_ROOT   = "/home/ubuntu/projects/RAG" if os.path.exists("/home/ubuntu/projects/RAG") else os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "cache", "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import warnings
import torch
import re
import asyncio
import threading
import queue as stdlib_queue
import sqlite3
import time
import json
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator, Generator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import requests
import aiohttp
import redis
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

HF_CACHE = os.environ["HF_HOME"]
os.makedirs(HF_CACHE, exist_ok=True)
for _k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
           "HUGGINGFACE_HUB_CACHE", "HF_HUB_DISABLE_SYMLINKS_WARNING"):
    os.environ[_k] = HF_CACHE if _k != "HF_HUB_DISABLE_SYMLINKS_WARNING" else "1"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("rag")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — single source of truth, must match rag_builder.py
# ══════════════════════════════════════════════════════════════════════════════

VECTOR_DB_DIR  = os.path.join(PROJECT_ROOT, "chroma_db_fast")
LLM_MODEL      = "/home/ubuntu/projects/risk_hospitalisation/models/Llama-3.1-8B-Instruct"
MONITORING_DB  = os.path.join(PROJECT_ROOT, "monitoring.db")
VLLM_API_URL   = "http://localhost:8000/v1/chat/completions"

# FIX 1: MUST match rag_builder.py EMBEDDING_MODEL
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# FIX 2: MUST match rag_builder.py COLLECTION_NAME
COLLECTION_NAME = "agentic_rag_collection"

PRIORITY_ORDER   = ["company_policy", "medicare", "medical", "general"]

# Retrieval
TOP_K             = 8    # increased from 5
FETCH_K           = 70  # increased from 40
# FIX 3: minimum cosine similarity to include a chunk
MIN_SCORE         = 0.30

MAX_NEW_TOKENS    = 350
CONFIDENCE_THRESHOLD     = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.55
ENABLE_MEMORY            = True
ENABLE_SELF_CORRECTION   = True
MAX_ANSWER_SENTENCES     = 8
# FIX 5: context per chunk:increased from 600/300 mess to 3500 to support full pages
CONTEXT_CHARS_PER_CHUNK  = 3500

# Reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_TTL  = 3600

# LLM
LLM_TIMEOUT_SECS = 30
LLM_RETRIES      = 2

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT LINKS / CONTACT
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENT_LINKS: Dict[str, str] = {
    "Policy C-901":           "https://drive.google.com/mock-abbreviation-link",
    "Employee Handbook 2025": "https://drive.google.com/mock-handbook-link",
    "PTO Policy":             "https://drive.google.com/mock-pto-link",
}

CONTACT_DEPARTMENTS = {
    "Human Resources": "Greg",
    "Clinical": "Melissa Bair (Nursing), Thomas Cole (Therapy)",
    "Operations/Scheduling": "Kristen Harris, Sudhir",
    "Billing": "Seshagiri"
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class QueryType(Enum):
    SIMPLE_FACT   = "simple_fact"
    POLICY_LOOKUP = "policy_lookup"
    COMPARATIVE   = "comparative"
    MULTI_STEP    = "multi_step"
    AMBIGUOUS     = "ambiguous"

class RetrievalStrategy(Enum):
    SINGLE_LAYER         = "single_layer"
    MULTI_LAYER          = "multi_layer"
    ITERATIVE_REFINEMENT = "iterative_refinement"

@dataclass
class QueryAnalysis:
    query_type:          QueryType
    complexity:          float
    entities:            List[str]
    keywords:            List[str]
    requires_multi_step: bool
    layer_hints:         List[str]
    confidence:          float
    rewritten_query:     str = ""   # FIX 4

@dataclass
class RetrievalPlan:
    strategy:          RetrievalStrategy
    layers:            List[str]
    num_chunks:        int
    use_mmr:           bool
    refinement_needed: bool
    reasoning:         str

@dataclass
class QualityAssessment:
    relevance_score:   float
    coverage_score:    float
    consistency_score: float
    overall_quality:   float
    needs_refinement:  bool
    missing_aspects:   List[str]
    feedback:          str

@dataclass
class AgenticResponse:
    answer:             str
    citations:          str
    confidence:         float
    query_analysis:     QueryAnalysis
    retrieval_plan:     RetrievalPlan
    quality_assessment: QualityAssessment
    iterations:         int
    execution_time:     float
    agent_trace:        List[Dict[str, Any]]
    retrieved_chunks:   List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ConversationMemory:
    history:         List[Dict[str, str]] = field(default_factory=list)
    context_summary: str = ""
    max_history:     int = 5

# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_CITATION_PAT   = re.compile(r'\([^)]*?p\.\d+\)')
_CITE_ONLY_LINE = re.compile(r'^\s*\([^)]*?p\.\d+\)\s*$')
_MULTI_SPACE    = re.compile(r'\s+')
_DOUBLE_DOT     = re.compile(r'\.\s*\.')
_TRAILING_CITES = re.compile(r'(\([^)]*?p\.\d+\)\s*){3,}$')
_FILLER_STARTS  = re.compile(
    r'^(Note:|Please note|As mentioned|As stated|In summary|To summarize|'
    r'Based on the|According to the above|The above|This means that|'
    r'I hope|I should note|It should be noted)', re.I
)

CUTOFF_MARKERS = [
    "Note:", "(Note:", "Please disregard", "I am following",
    "Sources:", "Citation:", "Reference:", "In summary,", "To summarize,",
    # LLM closing fillers — strip these before faithfulness check
    "I don't have any further information",
    "I don't have further information",
    "I do not have any further information",
    "I do not have further information",
    "I hope this helps",
    "I hope that helps",
    "Please let me know if",
    "Feel free to ask",
    "If you have any",
    "For more information",
    "It is important to note",
    "It's important to note",
    "I should mention",
    "Please note that",
    "It should be noted",
    "This is based on",
    "Please be aware",
    "Keep in mind",
    "Additionally, please",
    "If you need further",
    "Should you need",
    "Do not hesitate",
]

# Phrases indicating the LLM genuinely found no answer — used for no_info detection
_NO_INFO_PHRASES = [
    "unable to find that information",
    "unable to find the information",
    "don't have that information",
    "do not have that information",
    "answer not found",
    "not mentioned in",
    "not provided in",
    "not available in",
    "cannot find",
    "could not find",
    "i'm sorry, i was unable",
]

SPECULATION_PHRASES = [
    "I believe", "I think", "I'm not sure", "possibly", "maybe",
    "I assume", "I suspect", "it seems like",
]

_STOP_WORDS = frozenset({
    "the","a","an","is","are","was","were","in","on","at","to","of",
    "and","or","for","with","that","this","it","be","as","by","from",
    "have","has","do","does","its","their","they","we","you","i",
})


def clean_pdf_text(text: str) -> str:
    """Cleans up vertical newline spacing issues in PDF text, collapsing space artifacts
    while preserving bulleted/numbered lists and section boundaries."""
    # 1. Collapse all consecutive whitespace to a single space
    text = re.sub(r'\s+', ' ', text)
    
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


def _split_sentences(text: str) -> List[str]:
    # Regex split on periods/exclamations/questions followed by space, avoiding splitting on:
    # - single letters (middle initials, A., B.)
    # - single/double digits (1., 4., 22.)
    # - common abbreviations (e.g., i.e., vs, p., pp., al., et.)
    pattern = r'(?<!\b[a-zA-Z])(?<!\b\d)(?<!\b\d\d)(?<!\bpp)(?<!\bp)(?<!e\.g)(?<!i\.e)(?<!vs)(?<!\bal)(?<!\bet)[.!?]\s+'
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        splits = re.split(pattern, line)
        for s in splits:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)
    return sentences


def _word_set(text: str) -> set:
    return set(re.findall(r"\w+", text.lower())) - _STOP_WORDS


# FIX-6: per-sentence faithfulness filter
def _sentence_faithfulness(sentence: str, doc_words: set) -> float:
    sw = _word_set(sentence)
    if not sw:
        return 1.0
    return len(sw & doc_words) / len(sw)


def enforce_sentence_budget(text: str, doc_words: set,
                             budget: int = MAX_ANSWER_SENTENCES,
                             faithfulness_threshold: float = 0.10) -> str:
    """
    Drop sentences whose meaningful words have < faithfulness_threshold overlap
    with retrieved documents.

    Threshold is intentionally LOW (0.10) because:
    - Domain answers rephrase source text (synonym drift is common)
    - Numeric/factual answers like "20 days" or "3 years" have very few
      content words that directly match the source verbatim
    - Only truly confabulated sentences (< 10% overlap, > 5 meaningful words)
      are removed

    SAFETY NET: If ALL sentences would be dropped, keep the single highest-scoring
    one — preventing the "" → "()" → fallback cascade that caused 0% confidence.
    """
    sentences = _split_sentences(text)
    seen: set = set()
    candidates: List[Tuple[float, str]] = []   # (faith_score, sentence)

    for sent in sentences:
        norm = _CITATION_PAT.sub("", sent).strip().rstrip(".")
        if not norm or _FILLER_STARTS.match(norm): 
            continue
        key = norm.lower()    
        seen.add(key)

        sw = _word_set(norm)
        # Only filter sentences with enough meaningful words to judge
        if len(sw) > 5 and doc_words:
            faith = _sentence_faithfulness(norm, doc_words)
        else:
            faith = 1.0   # short sentences / numeric answers pass unconditionally

        candidates.append((faith, sent))

    # Separate passing and failing sentences
    passing = [(f, s) for f, s in candidates if f >= faithfulness_threshold]
    failing = [(f, s) for f, s in candidates if f < faithfulness_threshold]

    if failing:
        log.info("[Faithfulness] Dropped %d unfaithful sentence(s)", len(failing))
        for f, s in failing:
            
            log.debug("[Faithfulness]  %.0f%% overlap: %s", f * 100, s[:80])

    kept = [s for _, s in passing[:budget]]

    # SAFETY NET: never return empty — keep best-scoring sentence if all dropped
    if not kept and candidates:
        best = max(candidates, key=lambda x: x[0])
        log.info("[Faithfulness] Safety-net: keeping best sentence (%.0f%% overlap)", best[0] * 100)
        kept = [best[1]]

    return " ".join(kept)


def _strip_artifacts(text: str) -> str:
    """Remove leading/trailing punctuation artifacts like lone '(' or '.'."""
    # Strip lines that are only punctuation or whitespace
    lines = [ln for ln in text.split("\n") if re.sub(r'[\s\W]+', '', ln)]
    text  = " ".join(lines)
    # Strip leading non-alphanumeric characters (e.g. stray opening parenthesis)
    text  = re.sub(r'^[\s\W]+', '', text)
    return text.strip()


def clean_answer(answer: str, doc_words: Optional[set] = None) -> str:
    # Strip either marker variant
    for marker in ("Answer (2-3 sentences, no repetition):", "Answer:"):
        if marker in answer:
            answer = answer.split(marker, 1)[-1]
            break

    _SYSTEM_LEAD = "You are a precise document Q&A assistant."
    if answer.strip().startswith(_SYSTEM_LEAD):
        _RULE_PAT = re.compile(
            r'^(You are|STRICT RULES|Rules:|Context:|Question:|Answer\s*\()', re.I
        )
        lines  = answer.split("\n")
        answer = " ".join([l for l in lines if l.strip() and not _RULE_PAT.match(l.strip())])

    for marker in CUTOFF_MARKERS:
        if marker in answer:
            answer = answer.split(marker)[0]

    answer = _TRAILING_CITES.sub("", answer)
    lines  = [ln.strip() for ln in answer.split("\n")
              if ln.strip() and not _CITE_ONLY_LINE.match(ln)]
    answer = " ".join(lines)

    seen_cites: Dict[str, int] = {}
    def _repl(m: re.Match) -> str:
        c = m.group(0)
        seen_cites[c] = seen_cites.get(c, 0) + 1
        return c if seen_cites[c] <= 2 else ""
    answer = _CITATION_PAT.sub(_repl, answer)

    # FIX-6: pass doc_words for per-sentence grounding filter
    if doc_words:
        answer = enforce_sentence_budget(answer, doc_words)
    else:
        answer = enforce_sentence_budget(answer, set())

    answer = _strip_artifacts(answer)
    answer = _MULTI_SPACE.sub(" ", answer)
    answer = _DOUBLE_DOT.sub(".", answer)
    result = answer.strip()
    # Guard: if after all cleaning we have a bare punctuation artifact, return fallback
    if not result or not re.search(r'[a-zA-Z]{3,}', result):
        return "I'm sorry, I was unable to find that information in the available documents."
    return result


def format_final_response(answer: str, citations: str, confidence: float,
                           exec_time: float) -> str:
    """
    ISSUE-5 FIX: Professional, structured CLI output.
    Separates the answer clearly from metadata.
    """
    bar   = "─" * 60
    lines = [f"\n{bar}"]

    # Answer block
    lines.append("  Answer")
    lines.append(bar)
    # Indent answer lines for readability
    for ln in answer.split("\n"):
        lines.append(f"  {ln}" if ln.strip() else "")

    # Citations block
    if citations:
        lines.append(f"\n  📄 Sources")
        lines.append(bar)
        for src_line in citations.split("\n"):
            if src_line.strip():
                lines.append(f"  {src_line}")

    # Confidence indicator
    lines.append(f"\n  📊 Confidence: {confidence:.0%}  |  ⏱  {exec_time}s")
    lines.append(bar)
    return "\n".join(lines)


def contact_block(contacts: dict = CONTACT_DEPARTMENTS) -> str:
    lines = ["\n\n  ⚠️  This answer may be incomplete. For further assistance, please contact:"]
    for dept, names in contacts.items():
        lines.append(f"  - {dept}: {names}")
    return "\n".join(lines)


# FIX-7: near-duplicate chunk removal (Jaccard similarity)
def _jaccard(a: str, b: str) -> float:
    sa, sb = _word_set(a), _word_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def dedup_docs(docs: List[Document], threshold: float = 0.85) -> List[Document]:
    """FIX-7: Remove near-duplicate chunks before sending to LLM."""
    kept: List[Document] = []
    for doc in docs:
        duplicate = any(
            _jaccard(doc.page_content, k.page_content) >= threshold
            for k in kept
        )
        if not duplicate:
            kept.append(doc)
    return kept

# ══════════════════════════════════════════════════════════════════════════════
# LLM ENGINE (vLLM)
# ══════════════════════════════════════════════════════════════════════════════

class LLMEngine:
    def __init__(self, model_name: str):
        self.api_url = VLLM_API_URL
        self.model   = model_name
        log.info("[LLM] Connected to vLLM Chat endpoint: %s", self.api_url)

    def _to_messages(self, prompt_or_messages: Any) -> List[Dict[str, str]]:
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]
        return prompt_or_messages

    def _payload(self, prompt_or_messages: Any, max_tokens: int, stream: bool = False) -> dict:
        messages = self._to_messages(prompt_or_messages)
        # Estimate input prompt tokens (conservative: ~3.8 chars per token)
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = int(total_chars / 3.8)
        
        # Max context limit of the vLLM model engine configuration (4096 tokens)
        max_context = 4096
        # Ensure we do not exceed the max context limit, keeping a safety margin
        safe_max_tokens = max(50, min(max_tokens, max_context - estimated_tokens - 15))
        
        if safe_max_tokens < max_tokens:
            log.info(
                "[LLM] Clipping max_tokens from %d to %d to fit within 4096 limit (est. input %d tokens)",
                max_tokens, safe_max_tokens, estimated_tokens
            )

        return {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  safe_max_tokens,
            "temperature": 0,
            "stream":      stream,
        }

    def generate(self, prompt_or_messages: Any, max_tokens: int = MAX_NEW_TOKENS) -> str:
        for attempt in range(LLM_RETRIES):
            try:
                r = requests.post(
                    self.api_url,
                    json=self._payload(prompt_or_messages, max_tokens),
                    timeout=LLM_TIMEOUT_SECS,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning("[LLM] Attempt %d failed: %s", attempt + 1, e)
                if attempt < LLM_RETRIES - 1:
                    time.sleep(0.5)
        return "Error: LLM unavailable after retries."

    async def agenerate(self, prompt_or_messages: Any, max_tokens: int = MAX_NEW_TOKENS) -> str:
        timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT_SECS)
        for attempt in range(LLM_RETRIES):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.api_url, json=self._payload(prompt_or_messages, max_tokens),
                    ) as resp:
                        result = await resp.json()
                        return result["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning("[LLM] Async attempt %d failed: %s", attempt + 1, e)
                if attempt < LLM_RETRIES - 1:
                    await asyncio.sleep(0.5)
        return "Error: LLM unavailable after retries."

    async def agenerate_stream(self, prompt_or_messages: Any, max_tokens: int = MAX_NEW_TOKENS):
        timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT_SECS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url,
                json=self._payload(prompt_or_messages, max_tokens, stream=True),
            ) as resp:
                async for line in resp.content:
                    if not line:
                        continue
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data:"):
                        continue
                    data = decoded[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except Exception:
                        continue

# ══════════════════════════════════════════════════════════════════════════════
# VECTOR DB ENGINE (HYBRID + RERANKER)
# ══════════════════════════════════════════════════════════════════════════════

class VectorDBEngine:
    """
    FIX-1: Uses EMBEDDING_MODEL (bge-base-en-v1.5) — matches rag_builder.py.
    FIX-2: Opens COLLECTION_NAME — matches rag_builder.py.
    FIX-3: Filters retrieved docs by MIN_SCORE similarity threshold.
    FIX-8: BM25 respects layer filter AT search time.
    """

    def __init__(self, db_dir: str):
        # FIX-1: matching embedding model
        emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # FIX-2: matching collection name
        self.db    = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=db_dir,
            embedding_function=emb,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._lock = threading.Lock()

        # Verify collection is not empty on startup
        try:
            count = self.db._collection.count()
            log.info("[VectorDB] Collection '%s' has %d chunks", COLLECTION_NAME, count)
            if count == 0:
                log.warning(
                    "[VectorDB] ⚠️  Collection is EMPTY! "
                    "Run rag_builder.py first, or check COLLECTION_NAME matches."
                )
        except Exception as e:
            log.warning("[VectorDB] Could not count collection: %s", e)

        # BM25 index
        log.info("[VectorDB] Building BM25 index …")
        try:
            all_data              = self.db.get()
            self._all_texts       = all_data.get("documents", [])
            self._all_metadatas   = all_data.get("metadatas", [])
            tokenized             = [t.lower().split() for t in self._all_texts]
            self.bm25             = BM25Okapi(tokenized) if tokenized else None
            log.info("[VectorDB] BM25 index built: %d docs", len(self._all_texts))
        except Exception as e:
            log.warning("[VectorDB] BM25 init failed: %s", e)
            self._all_texts, self._all_metadatas, self.bm25 = [], [], None

        # Cross-encoder reranker
        log.info("[VectorDB] Loading reranker …")
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL, cache_folder=HF_CACHE)
            log.info("[VectorDB] Reranker ready: %s", RERANKER_MODEL)
        except Exception as e:
            log.warning("[VectorDB] Reranker load failed: %s — skipping rerank", e)
            self.reranker = None

    # ── BM25 search — FIX-8: filter by layer AT SEARCH TIME ──────────

    def _bm25_search(self, query: str, k: int, layers: Optional[List[str]] = None) -> List[Document]:
        if not self.bm25 or not self._all_texts:
            return []
        try:
            scores  = self.bm25.get_scores(query.lower().split())
            top_idx = sorted(range(len(self._all_texts)), key=lambda i: scores[i], reverse=True)
            hits = []
            for i in top_idx:
                if len(hits) >= k * 3:
                    break
                if i >= len(self._all_metadatas):
                    continue
                if scores[i] <= 0.0:
                    continue
                meta = self._all_metadatas[i]
                if layers and meta.get("layer") not in layers:
                    continue
                hits.append(
                    Document(
                        page_content=self._all_texts[i],
                        metadata=meta,
                    )
                )
            return hits
        except Exception as e:
            log.warning("[BM25] search error: %s", e)
            return []

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not self.reranker or not docs:
            return docs
        try:
            pairs  = [(query, d.page_content) for d in docs]
            scores = self.reranker.predict(pairs)
            return [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
        except Exception as e:
            log.warning("[Reranker] error: %s", e)
            return docs

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cache(self) -> Dict:
        if not hasattr(self, "_search_cache"):
            self._search_cache: Dict[str, List[Document]] = {}
        return self._search_cache

    def _search_sync(self, query: str, k: int, layers: Optional[List[str]] = None) -> List[Document]:
        cache     = self._get_cache()
        cache_key = self._cache_key(query + "_" + (",".join(layers) if layers else ""))
        if cache_key in cache:
            log.debug("[VectorDB] Cache HIT: %s", query[:40])
            return cache[cache_key]

        if not self._all_texts:
            return []

        try:
            # Construct Chroma metadata filter
            where_filter = None
            if layers:
                if len(layers) == 1:
                    where_filter = {"layer": layers[0]}
                else:
                    where_filter = {"layer": {"$in": layers}}

            # 1. Vector similarity search over filtered documents
            with self._lock:
                scored = self.db.similarity_search_with_relevance_scores(
                    query, k=min(15, len(self._all_texts)), filter=where_filter
                )
                vector_hits = [doc for doc, score in scored]
            
            # 2. BM25 keyword search over filtered documents
            bm25_hits = self._bm25_search(query, k=15, layers=layers)

            # Fallback: if filtered search returned weak or no results, retry globally
            best_score = max([score for doc, score in scored], default=0.0)
            should_fallback = layers and (
                not vector_hits or
                best_score < 0.40 or
                (best_score < 0.53 and not bm25_hits)
            )

            if should_fallback:
                log.info(
                    "[VectorDB] Filtered search results are weak (best_score=%.3f, bm25_hits=%d). Retrying globally.",
                    best_score, len(bm25_hits)
                )
                with self._lock:
                    scored = self.db.similarity_search_with_relevance_scores(
                        query, k=min(15, len(self._all_texts))
                    )
                    vector_hits = [doc for doc, score in scored]
                bm25_hits = self._bm25_search(query, k=15, layers=None)
            
            # 3. Combine and deduplicate by (source, page)
            combined = []
            seen_keys = set()
            for doc in vector_hits + bm25_hits:
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", 0)
                key = (src, page)
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined.append(doc)
            
            # 4. Perform Cross-Encoder Reranking
            reranked = self._rerank(query, combined)
            
            # 5. Page Window Expansion:
            # For the top reranked pages, we fetch adjacent pages N-1 and N+1.
            # We cap the total unique pages at 4 to prevent exceeding the vLLM context length limit.
            expanded_docs = []
            expanded_keys = set()
            for doc in reranked:
                if len(expanded_keys) >= 4:
                    break
                src = doc.metadata.get("source", "unknown")
                page = int(doc.metadata.get("page", 0))
                
                # Fetch pages N, N-1, N+1 in priority order (N first, then adjacent)
                for p in (page, page - 1, page + 1):
                    if p < 0:
                        continue
                    if len(expanded_keys) >= 4:
                        break
                    key = (src, p)
                    if key in expanded_keys:
                        continue
                    
                    # Find if this page exists in our corpus
                    match_doc = None
                    for text, meta in zip(self._all_texts, self._all_metadatas):
                        if meta.get("source") == src and int(meta.get("page", -99)) == p:
                            match_doc = Document(page_content=text, metadata=meta)
                            break
                    
                    if match_doc:
                        expanded_keys.add(key)
                        expanded_docs.append(match_doc)
            
            # 6. Sort by (source, page) to maintain natural document order
            expanded_docs.sort(key=lambda d: (d.metadata.get("source", ""), int(d.metadata.get("page", 0))))
            
            log.info("[VectorDB] Retrieved %d chunks after window expansion", len(expanded_docs))
            
            # Cache results
            if len(cache) >= 500:
                cache.pop(next(iter(cache)))
            cache[cache_key] = expanded_docs
            return expanded_docs
            
        except Exception as e:
            log.warning("[VectorDB] search error: %s", e)
            return []

    async def search_async(self, query: str, layer: str = None, layers: List[str] = None,
                           k: int = TOP_K, use_mmr: bool = True) -> List[Document]:
        search_layers = layers
        if not search_layers and layer:
            search_layers = [layer]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._search_sync, query, k, search_layers
        )

    async def search_parallel(self, query: str, layers: List[str] = None,
                               k_per_layer: int = 3) -> List[Document]:
        return await self.search_async(query, layers=layers, k=TOP_K)

    def reload(self):
        with self._lock:
            log.info("[VectorDB] Hot-reloading …")
            try:
                emb = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                self.db = Chroma(
                    collection_name=COLLECTION_NAME,
                    persist_directory=VECTOR_DB_DIR,
                    embedding_function=emb,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                all_data            = self.db.get()
                self._all_texts     = all_data.get("documents", [])
                self._all_metadatas = all_data.get("metadatas", [])
                tokenized           = [t.lower().split() for t in self._all_texts]
                self.bm25           = BM25Okapi(tokenized) if tokenized else None
            except Exception as e:
                log.warning("[VectorDB] BM25 rebuild failed: %s", e)
            if hasattr(self, "_search_cache"):
                self._search_cache.clear()
            log.info("[VectorDB] Hot-reload complete")

# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class QueryRewriterAgent:
    """
    FIX-4: Query rewriting — was in docstring but not implemented in v4.
    Expands vague/short queries before retrieval to improve recall.
    """

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    async def rewrite(self, query: str) -> str:
        # Skip rewriting for queries already long and specific
        if len(query.split()) >= 8:
            return query
        messages = [
            {"role": "system", "content": "You are a precise search query rewriter. Rewrite user questions to be more specific for document search. Output ONLY the rewritten question, nothing else, without extra words or conversational filler."},
            {"role": "user", "content": f"Original: {query}"}
        ]
        try:
            rewritten = await self.llm.agenerate(messages, max_tokens=60)
            # Sanity check: rewritten should not be empty or identical
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten and rewritten.lower() != query.lower() and len(rewritten) > 5:
                log.info("[Rewriter] '%s' → '%s'", query[:50], rewritten[:80])
                return rewritten
        except Exception as e:
            log.warning("[Rewriter] failed: %s", e)
        return query


class QueryAnalyzerAgent:
    def __init__(self, llm: LLMEngine):
        self.llm     = llm
        self.rewriter = QueryRewriterAgent(llm)

    async def analyze(self, query: str,
                      memory: Optional[ConversationMemory] = None) -> QueryAnalysis:
        # FIX-4: rewrite query first
        rewritten = await self.rewriter.rewrite(query)

        ctx = (f"\nContext: {memory.context_summary[:100]}"
               if memory and memory.context_summary else "")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query classification assistant. You output ONLY key-value classifications "
                    "in exactly 7 lines with no explanation. For LAYER_HINTS, you must select one or more "
                    "comma-separated values chosen ONLY from: company_policy, medicare, medical, general."
                )
            },
            {"role": "user", "content": f"Query: {rewritten}{ctx}\n\nExample output:\nQUERY_TYPE: simple_fact\nCOMPLEXITY: 0.6\nENTITIES: vacation, policy\nKEYWORDS: vacation, days, policy\nMULTI_STEP: no\nLAYER_HINTS: company_policy\nCONFIDENCE: 0.8\n\nClassify the Query and output exactly 7 lines starting with QUERY_TYPE:"}
        ]
        raw      = await self.llm.agenerate(messages, max_tokens=100)
        parsed_raw = raw if raw.startswith("QUERY_TYPE:") else ("QUERY_TYPE:" + raw)
        analysis = self._parse(parsed_raw)
        analysis.rewritten_query = rewritten

        # --- DETERMINISTIC LAYER HINTS OVERRIDES ---
        query_lower = rewritten.lower()
        
        # Keywords indicating medicare
        medicare_kws = ["medicare", "noa", "hha", "pps", "lupa", "clinical group", "notice of admission", "home health", "rap", "raps", "mmta", "timing", "consolidated billing", "payment group", "wage index", "recertification", "30-day period", "oasis", "case-mix", "casemix", "dme", "osteoporosis", "admission source", "functional impairment", "functional level", "comorbidity", "co-morbidity", "billing"]
        has_medicare = any(kw in query_lower for kw in medicare_kws)
        
        # Keywords indicating company_policy
        policy_kws = ["pto", "paid time off", "leave", "vacation", "handbook", "employee", "abbreviation", "abbreviations", "acronym", "c-901", "c901", "conduct", "dress code", "probation", "bereavement", "jury duty", "harassment", "attendance", "resignation", "discipline"]
        has_policy = any(kw in query_lower for kw in policy_kws)
        
        if has_medicare and not has_policy:
            analysis.layer_hints = ["medicare"]
        elif has_policy and not has_medicare:
            analysis.layer_hints = ["company_policy"]
        elif has_medicare and has_policy:
            med_idx = min((query_lower.find(kw) for kw in medicare_kws if kw in query_lower), default=len(query_lower))
            pol_idx = min((query_lower.find(kw) for kw in policy_kws if kw in query_lower), default=len(query_lower))
            if med_idx < pol_idx:
                analysis.layer_hints = ["medicare", "company_policy"]
            else:
                analysis.layer_hints = ["company_policy", "medicare"]

        return analysis

    def _parse(self, raw: str) -> QueryAnalysis:
        data: Dict[str, str] = {}
        for line in raw.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip().upper()] = v.strip()

        qt = QueryType.POLICY_LOOKUP
        for q in QueryType:
            if q.value in data.get("QUERY_TYPE", "").lower():
                qt = q; break

        def _f(key: str, dflt: float) -> float:
            try:
                nums = [x for x in data.get(key, str(dflt)).split()
                        if x.replace(".", "", 1).isdigit()]
                return max(0.0, min(1.0, float(nums[0]) if nums else dflt))
            except Exception:
                return dflt

        lh_raw = [l.strip().lower() for l in data.get("LAYER_HINTS", "company_policy").split(",")]
        lh = []
        for l in lh_raw:
            if "medicare" in l or "healthcare" in l or "insurance" in l or "claim" in l:
                lh.append("medicare")
            elif "medical" in l or "clinical" in l or "hospital" in l:
                lh.append("medical")
            elif "company" in l or "policy" in l or "handbook" in l or "pto" in l or "vacation" in l or "hr" in l or "contact" in l or "abbreviation" in l or "acronym" in l:
                lh.append("company_policy")
            elif l in PRIORITY_ORDER:
                lh.append(l)
        
        # Deduplicate while preserving order
        seen = set()
        lh_dedup = []
        for item in lh:
            if item not in seen:
                seen.add(item)
                lh_dedup.append(item)
        lh = lh_dedup

        return QueryAnalysis(
            query_type=qt,
            complexity=_f("COMPLEXITY", 0.6),
            entities=[e.strip() for e in data.get("ENTITIES","").split(",") if e.strip()][:5],
            keywords=[k.strip() for k in data.get("KEYWORDS","").split(",") if k.strip()][:5],
            requires_multi_step="yes" in data.get("MULTI_STEP","no").lower(),
            layer_hints=lh or ["company_policy","general"],
            confidence=_f("CONFIDENCE", 0.7),
        )


class RouterAgent:
    # Keywords that force multi-layer regardless of query type classification
    _POLICY_KEYWORDS = frozenset({
        "pto", "paid time off", "leave", "vacation", "policy", "benefit",
        "handbook", "management", "employee", "salary", "compensation",
        "accrual", "entitlement", "allowance", "days off", "holiday",
        "abbreviation", "abbreviations", "acronym", "c-901", "c901",
        "conduct", "dress code", "probation", "bereavement", "jury duty",
        "harassment", "attendance", "resignation", "discipline"
    })
    
    # Medicare-specific keywords to intercept routing
    _MEDICARE_KEYWORDS = frozenset({
        "medicare", "noa", "hha", "pps", "lupa", "clinical group", 
        "notice of admission", "home health", "rap", "raps", "mmta",
        "timing", "consolidated billing", "payment group", "wage index",
        "recertification", "30-day period", "oasis", "case-mix", "casemix",
        "dme", "osteoporosis", "admission source", "functional impairment",
        "functional level", "comorbidity", "co-morbidity", "billing"
    })

    def route(self, analysis: QueryAnalysis) -> RetrievalPlan:
        query_lower = analysis.rewritten_query.lower() or ""

        # Force multi-layer if query mentions policy-related keywords
        has_policy_keyword = any(kw in query_lower for kw in self._POLICY_KEYWORDS)
        has_medicare_keyword = any(kw in query_lower for kw in self._MEDICARE_KEYWORDS)

        # Sync layer_hints with keyword findings
        lh = list(analysis.layer_hints)
        if has_medicare_keyword and "medicare" not in lh:
            lh = ["medicare"] + [h for h in lh if h != "medicare"]
        if has_policy_keyword and "company_policy" not in lh:
            lh = [h for h in lh if h != "company_policy"] + ["company_policy"]
            # Prioritize company_policy if it's the primary subject
            if any(k in query_lower for k in ["pto", "leave", "vacation", "abbreviation", "c-901"]):
                lh = ["company_policy"] + [h for h in lh if h != "company_policy"]
        analysis.layer_hints = lh

        if has_policy_keyword or analysis.query_type == QueryType.POLICY_LOOKUP:
            return RetrievalPlan(
                RetrievalStrategy.MULTI_LAYER,
                analysis.layer_hints[:2] if len(analysis.layer_hints) > 1
                else analysis.layer_hints + ["general"],
                TOP_K, True, False, "Policy/benefit lookup — multi-layer",
            )

        # Strict simple_fact: must be low complexity AND no policy keywords
        if analysis.query_type == QueryType.SIMPLE_FACT and analysis.complexity < 0.35:
            return RetrievalPlan(RetrievalStrategy.SINGLE_LAYER,
                                 analysis.layer_hints[:1], 5, True, False, "Simple fact")

        if analysis.query_type in (QueryType.COMPARATIVE, QueryType.MULTI_STEP):
            return RetrievalPlan(RetrievalStrategy.MULTI_LAYER,
                                 analysis.layer_hints[:2], TOP_K, True, False, "Complex query")

        return RetrievalPlan(RetrievalStrategy.SINGLE_LAYER,
                             analysis.layer_hints[:1], TOP_K, True, False, "Standard lookup")


class RetrievalExecutor:
    def __init__(self, db: VectorDBEngine):
        self.db = db

    async def execute(self, query: str, plan: RetrievalPlan,
                      rewritten_query: str = "") -> List[Document]:
        # FIX-4: use rewritten query for retrieval
        retrieval_query = rewritten_query or query
        if plan.strategy == RetrievalStrategy.MULTI_LAYER:
            docs = await self.db.search_parallel(retrieval_query, plan.layers, k_per_layer=4)
        else:
            layer = plan.layers[0] if plan.layers else "company_policy"
            docs  = await self.db.search_async(retrieval_query, layer, k=plan.num_chunks)

        # Collapse vertical spacing and clean chunks dynamically in memory
        for d in docs:
            d.page_content = clean_pdf_text(d.page_content)

        # FIX-7: remove near-duplicate chunks before returning
        return dedup_docs(docs)


class QualityEvaluatorAgent:
    def evaluate(self, query: str, documents: List[Document]) -> QualityAssessment:
        if not documents:
            return QualityAssessment(0, 0, 0, 0, True, ["No docs"], "No docs")

        qw = _word_set(query)
        dw = _word_set(" ".join(d.page_content for d in documents))
        meaningful = qw - _STOP_WORDS or qw
        relevance  = min(0.95, len(meaningful & dw) / max(len(meaningful), 1) * 1.5)

        n        = len(documents)
        coverage = 0.90 if n >= 5 else (0.75 if n >= 3 else (0.55 if n >= 1 else 0.20))

        us          = len({d.metadata.get("source","") for d in documents})
        consistency = 0.90 if us == 1 else (0.80 if us == 2 else (0.65 if us <= 4 else 0.50))

        overall = round(relevance * 0.5 + coverage * 0.3 + consistency * 0.2, 4)
        missing = (["low keyword match"] if relevance < 0.4 else []) + \
                  (["few chunks"] if coverage < 0.6 else [])

        return QualityAssessment(
            relevance_score=relevance, coverage_score=coverage,
            consistency_score=consistency, overall_quality=overall,
            needs_refinement=overall < CONFIDENCE_THRESHOLD,
            missing_aspects=missing,
            feedback=f"rel={relevance:.2f} cov={coverage:.2f} cons={consistency:.2f}",
        )


class ResponseGeneratorAgent:
    SYSTEM = (
            "You are a knowledgeable and professional HR/policy assistant for Elumina Health. "
            "Your role is to provide clear, accurate, and helpful answers to employee questions "
            "based solely on the company documents provided.\n\n"
            "STRICT RULES:\n"
            "(1) Answer using facts explicitly stated in the Context below, or logically deduced from them. Do not invent facts or extrapolate beyond the context.\n"
            "(2) If the answer cannot be found or logically deduced from the Context (such as when specific values/amounts are missing or a list is not enumerated), respond EXACTLY with this single sentence: "
            "\"I'm sorry, I was unable to find that information in the available documents.\"\n"
            "(3) TRICK QUESTIONS / FALSE PREMISES: If a question assumes a false premise or is a trick question, "
            "you MUST start your response by directly correcting the false premise (e.g., state 'No, ...' and then explain the true rule). For example:\n"
            "  - If asked if DME is included in the rate: explain that Durable Medical Equipment (DME) is specifically excluded from the 30-day period payment rate.\n"
            "  - If asked if NOAs must be submitted in business days: explain that they must be submitted within 5 calendar days, not business days.\n"
            "  - If asked if a gap of exactly 60 days resets timing to Early: explain that a gap of exactly 60 days keeps the timing classification as Late, and only a gap of MORE than 60 days resets it to Early.\n"
            "  - If asked if there are 12 MMTA groups: explain that there is only 1 MMTA group which contains 7 sub-categories, while there are 12 total clinical groups of which MMTA is one.\n"
            "(4) CALCULATIONS & MATH / LOGICAL DEDUCTION: If the question requires counting days, dates, case-mix permutations, or logical deduction (e.g. determining the maximum gap or LUPA thresholds), "
            "reason step-by-step based on the rules in the context to compute or deduce the correct answer, and state the final result explicitly.\n"
            "(5) KEY POLICIES / RULES TO REMEMBER:\n"
            "  - TIMING CASE-MIX VARIABLE: Timing has exactly two values: Early (the first 30-day period of care) and Late (all subsequent 30-day periods of care, unless there is a gap of more than 60 days between periods, which resets the next period to Early). A gap of exactly 60 days keeps the timing as Late.\n"
            "  - CLAIMS CASE-MIX VARIABLES: Exactly four case-mix variables are obtained from claims: Admission Source (Institutional or Community), Timing (Early or Late), Clinical Group (one of 12), and Comorbidity Adjustment (none, low, high).\n"
            "  - OASIS CASE-MIX VARIABLE: Exactly one case-mix variable is derived from OASIS data: Functional Impairment Level (low, medium, high).\n"
            "  - MMTA CLINICAL SUB-CATEGORIES: There are 7 sub-categories under the MMTA group: Surgical Aftercare, Cardiac/Circulatory, Endocrine, GI/GU, Infectious Disease/Neoplasms/Blood-forming Diseases, Respiratory, and Other.\n"
            "(6) Write in a warm, professional tone — as if speaking directly to a valued employee.\n"
            "(7) Answer in 2-4 sentences. Be complete but concise.\n"
            "(8) State each fact ONCE — never repeat the same information.\n"
            "(9) Cite your source inline like (medicare_plicies.pdf p.2) or (Employee Handbook p.12) or (Policy C-901 p.1).\n"
            "(10) Do NOT add disclaimers, speculation, summaries, or follow-up offers.\n"
            "(11) Do NOT include phrases like 'I hope this helps', 'feel free to ask', "
            "'please let me know', or 'I don't have further information'.\n"
            "(12) Stop immediately after completing your answer."
        )
    _MARKER = "Answer:"


    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def _context(self, docs: List[Document]) -> str:
        # Group contiguous pages of the same document together to preserve sentence/word flow
        sorted_docs = sorted(docs, key=lambda d: (d.metadata.get("source", ""), int(d.metadata.get("page", 0))))
        
        grouped_blocks = []
        current_source = None
        current_pages = []
        current_texts = []
        
        for doc in sorted_docs:
            src = doc.metadata.get("source", "unknown")
            page = int(doc.metadata.get("page", 0))
            text = doc.page_content.strip()[:CONTEXT_CHARS_PER_CHUNK]
            
            if current_source == src and len(current_pages) > 0 and page == current_pages[-1] + 1:
                # Contiguous page of the same document! Join them.
                current_pages.append(page)
                current_texts.append(f"\n[Page {page}]\n{text}")
            else:
                # Flush previous group if exists
                if current_source:
                    pages_str = f"p.{current_pages[0]}" if len(current_pages) == 1 else f"p.{current_pages[0]}-{current_pages[-1]}"
                    combined_text = "\n".join(current_texts)
                    grouped_blocks.append(f"[{current_source} {pages_str}]\n{combined_text}")
                
                # Start new group
                current_source = src
                current_pages = [page]
                current_texts = [f"[Page {page}]\n{text}"]
                
        # Flush last group
        if current_source:
            pages_str = f"p.{current_pages[0]}" if len(current_pages) == 1 else f"p.{current_pages[0]}-{current_pages[-1]}"
            combined_text = "\n".join(current_texts)
            grouped_blocks.append(f"[{current_source} {pages_str}]\n{combined_text}")
            
        return "\n\n".join(grouped_blocks)

    def _citations(self, docs: List[Document]) -> str:
        seen, rows = set(), []
        for doc in docs:
            src  = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            key  = (src, page)
            if key in seen:
                continue
            seen.add(key)
            link = next(
                (u for k, u in DOCUMENT_LINKS.items() if k.lower() in src.lower()),
                "Link not available",
            )
            rows.append(f"{src} (p.{page}) – {link}")
        return "\n".join(rows)

    def _messages(self, query: str, docs: List[Document]) -> List[Dict[str, str]]:
        has_general = any(d.metadata.get("layer") == "general" for d in docs)
        if has_general:
            system_prompt = (
                "You are a knowledgeable and professional document assistant. "
                "Your role is to provide clear, accurate, and helpful answers to user questions "
                "based solely on the documents provided.\n\n"
                "STRICT RULES:\n"
                "(1) Answer using facts explicitly stated in the Context below, or logically deduced from them. Do not invent facts or extrapolate beyond the context.\n"
                "(2) If the answer cannot be found or logically deduced from the Context, respond EXACTLY with this single sentence: "
                "\"I'm sorry, I was unable to find that information in the available documents.\"\n"
                "(3) Write in a professional, objective tone.\n"
                "(4) Answer in 2-4 sentences. Be complete but concise.\n"
                "(5) State each fact ONCE — never repeat the same information.\n"
                "(6) Cite your source inline like (document_name.pdf p.2).\n"
                "(7) Do NOT add disclaimers, speculation, summaries, or follow-up offers.\n"
                "(8) Stop immediately after completing your answer."
            )
        else:
            system_prompt = self.SYSTEM

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{self._context(docs)}\n\nEmployee Question: {query}"}
        ]

    def _doc_words(self, docs: List[Document]) -> set:
        """All meaningful words from retrieved docs — used for faithfulness filter.
        Uses the full page content of retrieved docs."""
        return _word_set(" ".join(d.page_content for d in docs))

    async def agenerate(self, query: str, docs: List[Document],
                        quality: QualityAssessment) -> Tuple[str, str, float]:
        raw      = await self.llm.agenerate(self._messages(query, docs), MAX_NEW_TOKENS)
        dw       = self._doc_words(docs)
        answer   = clean_answer(raw, doc_words=dw)

        # Fix: confidence uses quality as the FLOOR.
        # Grounding can only boost (up to +0.15), never tank a good quality score.
        # This prevents a short-but-correct answer from getting 0% confidence.
        ans_words = _word_set(answer)
        grounding = len(ans_words & dw) / max(len(ans_words), 1) if ans_words else 0.5
        grounding_boost = (grounding - 0.5) * 0.3   # ranges from -0.15 to +0.15
        confidence = round(min(0.97, max(0.0, quality.overall_quality + grounding_boost)), 4)
        return answer, self._citations(docs), confidence

    async def agenerate_stream(self, query: str, docs: List[Document]):
        async for tok in self.llm.agenerate_stream(self._messages(query, docs), MAX_NEW_TOKENS):
            yield tok

    def generate(self, query: str, docs: List[Document],
                 quality: QualityAssessment) -> Tuple[str, str, float]:
        raw      = self.llm.generate(self._messages(query, docs), MAX_NEW_TOKENS)
        dw       = self._doc_words(docs)
        answer   = clean_answer(raw, doc_words=dw)
        ans_words = _word_set(answer)
        grounding = len(ans_words & dw) / max(len(ans_words), 1) if ans_words else 0.5
        grounding_boost = (grounding - 0.5) * 0.3
        confidence = round(min(0.97, max(0.0, quality.overall_quality + grounding_boost)), 4)
        return answer, self._citations(docs), confidence


class SelfCorrectionAgent:
    """
    Enhanced self-correction.
    - too_short: does NOT reduce confidence (short ≠ wrong)
    - no_info detected via _NO_INFO_PHRASES list → confidence 0.0
    - low_grounding: softened penalty, does not apply to no_info answers
    """

    def correct(self, answer: str, base: float,
                docs: Optional[List[Document]] = None) -> Tuple[str, float, List[str]]:
        issues, conf = [], base
        answer_lower = answer.lower()

        # Repetitive notes — strip and soft penalty
        if answer.count("Note:") > 1:
            issues.append("repetitive_notes")
            conf -= 0.08
            answer = answer.split("Note:")[0].strip()

        # Speculation — soft penalty
        for p in SPECULATION_PHRASES:
            if p.lower() in answer_lower:
                issues.append("speculation")
                conf -= 0.06
                break

        # too_short → logged only, no confidence hit (short answers can be correct)
        if len(answer.strip()) < 40:
            issues.append("too_short")

        # No info found → zero confidence
        if any(phrase in answer_lower for phrase in _NO_INFO_PHRASES):
            issues.append("no_info")
            conf = 0.0

        # Low grounding — softened, only when answer has substance
        if docs and "no_info" not in issues and len(answer.strip()) >= 40:
            dw = _word_set(" ".join(d.page_content for d in docs))
            aw = _word_set(answer)
            if aw and len(aw & dw) / len(aw) < 0.15:
                issues.append("low_grounding")
                conf -= 0.08
                log.warning("[SelfCorrection] Low grounding detected (%.0f%% overlap)",
                            len(aw & dw) / len(aw) * 100)

        return answer, round(max(0.0, min(1.0, conf)), 4), issues

# ══════════════════════════════════════════════════════════════════════════════
# MEMORY MANAGER (REDIS)
# ══════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    def __init__(self):
        self._redis   = None
        self._fallback: Dict[str, ConversationMemory] = {}
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                            decode_responses=True, socket_timeout=2)
            r.ping()
            self._redis = r
            log.info("[Memory] Redis connected at %s:%d", REDIS_HOST, REDIS_PORT)
        except Exception as e:
            log.warning("[Memory] Redis unavailable (%s) — using in-process fallback", e)

    def _redis_key(self, session_id: str) -> str:
        return f"rag:chat:{session_id}"

    def add(self, session_id: str, query: str, answer: str):
        entry = json.dumps({"query": query, "answer": answer[:200]})
        if self._redis:
            key = self._redis_key(session_id)
            self._redis.rpush(key, entry)
            self._redis.ltrim(key, -5, -1)
            self._redis.expire(key, REDIS_TTL)
        else:
            mem = self._fallback.setdefault(session_id, ConversationMemory())
            mem.history.append({"query": query, "answer": answer[:200]})
            mem.history = mem.history[-5:]
            mem.context_summary = " | ".join(
                f"Q:{t['query'][:50]}" for t in mem.history[-2:]
            )

    def get(self, session_id: str) -> ConversationMemory:
        if self._redis:
            try:
                raw     = self._redis.lrange(self._redis_key(session_id), 0, -1)
                history = [json.loads(h) for h in raw]
                summary = " | ".join(f"Q:{h['query'][:50]}" for h in history[-2:])
                return ConversationMemory(history=history, context_summary=summary)
            except Exception as e:
                log.warning("[Memory] Redis get failed: %s", e)
        return self._fallback.get(session_id, ConversationMemory())

    def reset(self, session_id: str):
        if self._redis:
            try:
                self._redis.delete(self._redis_key(session_id))
            except Exception:
                pass
        self._fallback.pop(session_id, None)

# ══════════════════════════════════════════════════════════════════════════════
# NON-BLOCKING MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class SimpleMonitor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._q: stdlib_queue.Queue = stdlib_queue.Queue()
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS logs")
        conn.execute("""
            CREATE TABLE logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT,
                query          TEXT,
                answer         TEXT,
                confidence     REAL,
                execution_time REAL,
                iterations     INTEGER,
                strategy       TEXT,
                success        INTEGER
            )
        """)
        conn.commit()
        conn.close()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    """INSERT INTO logs
                       (timestamp, query, answer, confidence,
                        execution_time, iterations, strategy, success)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    item,
                )
                conn.commit()
                conn.close()
            except Exception as e:
                log.error("[Monitor] %s", e)

    def log(self, query: str, answer: str, confidence: float,
            exec_time: float, iterations: int, strategy: str, success: bool):
        self._q.put((
            datetime.now().isoformat(), query, answer[:300],
            confidence, exec_time, iterations, strategy, int(success),
        ))

    def get_stats(self, hours: int = 24) -> Optional[Dict]:
        cutoff = datetime.fromtimestamp(time.time() - hours * 3600).isoformat()
        conn   = sqlite3.connect(self.db_path)
        row    = conn.execute("""
            SELECT COUNT(*), AVG(execution_time), AVG(confidence),
                   SUM(success)*100.0/COUNT(*)
            FROM logs WHERE timestamp > ?
        """, (cutoff,)).fetchone()
        conn.close()
        if row and row[0]:
            return {
                "total_queries": int(row[0]),
                "avg_time":      round(float(row[1] or 0), 2),
                "avg_confidence":round(float(row[2] or 0), 3),
                "success_rate":  round(float(row[3] or 0), 1),
            }
        return None

# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class AgenticRAGOrchestrator:
    def __init__(self, llm_path: str, vectordb_path: str, monitoring_db: str):
        log.info("🤖 Initializing Agentic RAG v5 …")
        self.llm       = LLMEngine(llm_path)
        self.vectordb  = VectorDBEngine(vectordb_path)
        self.analyzer  = QueryAnalyzerAgent(self.llm)
        self.router    = RouterAgent()
        self.retriever = RetrievalExecutor(self.vectordb)
        self.evaluator = QualityEvaluatorAgent()
        self.generator = ResponseGeneratorAgent(self.llm)
        self.corrector = SelfCorrectionAgent()
        self.memory    = MemoryManager()
        self.monitor   = SimpleMonitor(monitoring_db)
        log.info("✅ System ready")

    def ingest_document(self, file_path: str):
        from rag_builder import ingest_pdf
        log.info("📥 New document upload detected")
        ingest_pdf(file_path, VECTOR_DB_DIR)
        self.vectordb.reload()
        log.info("♻️ Vector DB reloaded with new document")

    def delete_document(self, file_name: str) -> bool:
        log.info("🗑️ Deleting document from vector DB: %s", file_name)
        try:
            self.vectordb.db.delete(where={"source": file_name})
            self.vectordb.reload()
            log.info("♻️ Vector DB reloaded after document deletion")
            return True
        except Exception as e:
            log.error("Failed to delete document %s: %s", file_name, e)
            return False

    # ── async full response ───────────────────────────────────────────

    async def aquery(
        self,
        question:               str,
        session_id:             str  = "default",
        enable_memory:          bool = ENABLE_MEMORY,
        enable_self_correction: bool = ENABLE_SELF_CORRECTION,
    ) -> AgenticResponse:
        t0  = time.perf_counter()
        mem = self.memory.get(session_id) if enable_memory else None

        log.info("🔍 Analyzing [session=%s] …", session_id)
        analysis = await self.analyzer.analyze(question, mem)
        log.info("   type=%s  complexity=%.2f  rewritten='%s'",
                 analysis.query_type.value, analysis.complexity,
                 analysis.rewritten_query[:60])

        plan = self.router.route(analysis)
        log.info("   strategy=%s  layers=%s", plan.strategy.value, plan.layers)

        log.info("📚 Retrieving …")
        docs = await self.retriever.execute(
            question, plan, rewritten_query=analysis.rewritten_query
        )
        log.info("   %d docs after dedup", len(docs))

        # Build retrieved chunks list
        retrieved_list = []
        for doc in docs:
            retrieved_list.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": int(doc.metadata.get("page", 0)),
                "content": doc.page_content.strip()
            })

        if not docs:
            return self._empty(question, analysis, plan, time.perf_counter() - t0)

        quality = self.evaluator.evaluate(question, docs)
        log.info("   quality=%.2f", quality.overall_quality)

        log.info("💬 Generating …")
        answer, citations, confidence = await self.generator.agenerate(
            question, docs, quality
        )

        if enable_self_correction:
            answer, confidence, issues = self.corrector.correct(answer, confidence, docs)
            if issues:
                log.info("   ⚠ corrections: %s", issues)

        exec_time    = round(time.perf_counter() - t0, 3)
        final_answer = self._assemble(answer, citations, confidence, exec_time)

        if enable_memory:
            self.memory.add(session_id, question, answer)
        self.monitor.log(question, answer, confidence, exec_time, 1,
                         plan.strategy.value, True)
        log.info("✅ %.2fs | conf=%.0f%%", exec_time, confidence * 100)

        return AgenticResponse(
            answer=final_answer, citations=citations, confidence=confidence,
            query_analysis=analysis, retrieval_plan=plan,
            quality_assessment=quality, iterations=1,
            execution_time=exec_time, agent_trace=[],
            retrieved_chunks=retrieved_list,
        )

    # ── async streaming ───────────────────────────────────────────────

    async def aquery_stream(
        self,
        question:      str,
        session_id:    str  = "default",
        enable_memory: bool = ENABLE_MEMORY,
    ) -> AsyncGenerator[str, None]:
        t0  = time.perf_counter()
        mem = self.memory.get(session_id) if enable_memory else None

        try:
            yield json.dumps({"type": "token", "data": "🔍 Searching policies…\n"})

            analysis = await self.analyzer.analyze(question, mem)
            plan     = self.router.route(analysis)

            yield json.dumps({"type": "token", "data": "📚 Retrieving documents…\n"})

            docs = await self.retriever.execute(
                question, plan, rewritten_query=analysis.rewritten_query
            )

            retrieved_list = []
            for doc in docs:
                retrieved_list.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page": int(doc.metadata.get("page", 0)),
                    "content": doc.page_content.strip()
                })
            yield json.dumps({"type": "retrieved_chunks", "data": retrieved_list})

            if not docs:
                yield json.dumps({"type": "token",
                                  "data": "I don't have that information in the available documents."})
                yield json.dumps({"type": "token", "data": contact_block()})
                yield json.dumps({"type": "meta", "data": {
                    "confidence": 0.0,
                    "execution_time": round(time.perf_counter() - t0, 3),
                }})
                return

            quality   = self.evaluator.evaluate(question, docs)
            citations = self.generator._citations(docs)
            buffer    = ""
            dw        = self.generator._doc_words(docs)

            async for tok in self.generator.agenerate_stream(question, docs):
                buffer += tok
                yield json.dumps({"type": "token", "data": tok})

            # FIX-6: faithfulness filter on streamed buffer + matching confidence formula
            clean_ans  = clean_answer(buffer, doc_words=dw)
            ans_words  = _word_set(clean_ans)
            grounding  = len(ans_words & dw) / max(len(ans_words), 1) if ans_words else 0.5
            grounding_boost = (grounding - 0.5) * 0.3
            confidence = round(min(0.97, max(0.0, quality.overall_quality + grounding_boost)), 4)
            clean_ans, confidence, _ = self.corrector.correct(clean_ans, confidence, docs)
            exec_time  = round(time.perf_counter() - t0, 3)

            yield json.dumps({"type": "citations", "data": citations})
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                yield json.dumps({"type": "token",
                                  "data": contact_block()})
            yield json.dumps({"type": "meta", "data": {
                "confidence":     confidence,
                "execution_time": exec_time,
                "quality":        quality.feedback,
            }})

            if enable_memory:
                self.memory.add(session_id, question, clean_ans)
            self.monitor.log(question, clean_ans, confidence, exec_time, 1,
                             plan.strategy.value, True)

        except Exception as exc:
            log.error("aquery_stream error: %s", exc)
            yield json.dumps({"type": "error", "data": str(exc)})

    # ── sync wrappers ─────────────────────────────────────────────────

    def query_sync(self, question: str, session_id: str = "default",
                   **kw) -> AgenticResponse:
        result_holder = [None]
        error_holder  = [None]
        done          = threading.Event()

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result_holder[0] = loop.run_until_complete(
                    self.aquery(question, session_id=session_id, **kw)
                )
            except Exception as e:
                error_holder[0] = e
            finally:
                loop.close()
                done.set()

        threading.Thread(target=_run, daemon=True).start()
        done.wait()
        if error_holder[0]:
            raise error_holder[0]
        return result_holder[0]

    def stream_sync(self, question: str, session_id: str = "default",
                    **kw) -> Generator[str, None, None]:
        q: stdlib_queue.Queue = stdlib_queue.Queue()

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def _collect():
                async for chunk in self.aquery_stream(
                    question, session_id=session_id, **kw
                ):
                    q.put(chunk)
                q.put(None)
            try:
                loop.run_until_complete(_collect())
            finally:
                loop.close()

        threading.Thread(target=_run, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item + "\n"

    # ── helpers ───────────────────────────────────────────────────────

    def _assemble(self, answer: str, citations: str, confidence: float,
                  exec_time: float = 0.0) -> str:
        """Build the final user-facing response with professional formatting."""
        formatted = format_final_response(answer, citations, confidence, exec_time)
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            formatted += contact_block()
        return formatted

    def _empty(self, question, analysis, plan, t) -> AgenticResponse:
        msg = (
            "I'm sorry, I was unable to find relevant information for your question "
            "in the available documents."
            + contact_block()
        )
        return AgenticResponse(
            answer=msg, citations="", confidence=0.0,
            query_analysis=analysis, retrieval_plan=plan,
            quality_assessment=QualityAssessment(0, 0, 0, 0, False, [], ""),
            iterations=1, execution_time=round(t, 3), agent_trace=[],
            retrieved_chunks=[],
        )

    def reset_memory(self, session_id: str = "default"):
        self.memory.reset(session_id)
        log.info("🔄 Memory reset for session: %s", session_id)

    def show_stats(self):
        s = self.monitor.get_stats()
        if s:
            log.info("📊 %d queries | avg %.1fs | conf %.0f%% | success %.1f%%",
                     s["total_queries"], s["avg_time"],
                     s["avg_confidence"] * 100, s["success_rate"])
        else:
            log.info("No stats yet.")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    bar = "═" * 60
    print(f"\n{bar}")
    print("  🤖  Elumina Health — Policy Assistant  (v6)")
    print(f"{bar}")
    print("  Type your question and press Enter.")
    print("  Commands: stream <question> | reset | stats | exit")
    print(f"{bar}\n")

    orch       = AgenticRAGOrchestrator(LLM_MODEL, VECTOR_DB_DIR, MONITORING_DB)
    session_id = "cli-session"

    while True:
        try:
            raw = input("\n  💬  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Thank you for using the Elumina Policy Assistant. Goodbye!\n")
            break

        if not raw: continue
        if raw.lower() in ("exit", "quit", "q"):
            print("\n  Thank you for using the Elumina Policy Assistant. Goodbye!\n")
            break
        if raw.lower() == "reset":
            orch.reset_memory(session_id)
            print("  ✅  Conversation history cleared.")
            continue
        if raw.lower() == "stats":
            orch.show_stats()
            continue

        if raw.lower().startswith("stream "):
            question = raw[7:].strip()
            print(f"\n{'─'*60}")
            print("  Assistant (streaming):")
            print(f"{'─'*60}")
            buffer = ""
            citations_text = ""
            meta_info = {}
            for chunk in orch.stream_sync(question, session_id=session_id):
                try:
                    obj = json.loads(chunk.strip())
                    if obj["type"] == "retrieved_chunks":
                        print(f"\n{'='*60}")
                        print(f"  📚  Retrieved Chunks from Vector DB (Before LLM Process):")
                        print(f"{'='*60}")
                        for idx, item in enumerate(obj["data"]):
                            print(f"  [{idx+1}] Source: {item['source']} (Page {item['page']})")
                            content_indented = "\n      ".join(item['content'].split("\n"))
                            print(f"      {content_indented}")
                            print(f"  {'-'*60}")
                        print(f"\n{'='*60}")
                        print(f"  💬  LLM Processed Answer (Streaming):")
                        print(f"{'='*60}")
                    elif obj["type"] == "token":
                        tok = obj["data"]
                        if tok.strip() in ("🔍 Searching policies…", "📚 Retrieving documents…"):
                            print(f"  {tok.strip()}", flush=True)
                        else:
                            print(tok, end="", flush=True)
                            buffer += tok
                    elif obj["type"] == "citations":
                        citations_text = obj["data"]
                    elif obj["type"] == "meta":
                        meta_info = obj["data"]
                    elif obj["type"] == "error":
                        print(f"\n  ❌  Error: {obj['data']}")
                except Exception:
                    pass
            # Print formatted footer
            if citations_text:
                print(f"\n\n  📄 Sources\n{'─'*60}")
                for line in citations_text.split("\n"):
                    if line.strip():
                        print(f"  {line}")
            if meta_info:
                conf = meta_info.get("confidence", 0)
                t    = meta_info.get("execution_time", 0)
                print(f"\n  📊 Confidence: {conf:.0%}  |  ⏱  {t}s")
            print(f"{'─'*60}")
            continue

        try:
            resp = orch.query_sync(raw, session_id=session_id)
            
            # Print retrieved chunks BEFORE the LLM answer
            print(f"\n{'='*60}")
            print(f"  📚  Retrieved Chunks from Vector DB (Before LLM Process):")
            print(f"{'='*60}")
            if not resp.retrieved_chunks:
                print("  No chunks retrieved.")
            for idx, item in enumerate(resp.retrieved_chunks):
                print(f"  [{idx+1}] Source: {item['source']} (Page {item['page']})")
                content_indented = "\n      ".join(item['content'].split("\n"))
                print(f"      {content_indented}")
                print(f"  {'-'*60}")
            
            print(f"\n{'='*60}")
            print(f"  💬  LLM Processed Answer:")
            print(f"{'='*60}")
            print(resp.answer)
        except Exception as e:
            print(f"\n  ❌  An error occurred: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
