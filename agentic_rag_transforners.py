#!/usr/bin/env python3
"""
Agentic RAG System - v3 (Async + Streaming + Concise)
======================================================
ENHANCEMENTS v3:
- Full async pipeline (asyncio) — parallelised where possible
- Token streaming from LLM → sub-5s perceived latency
- Concise response engine — hard deduplication + sentence budget
- Parallel retrieval across layers (asyncio.gather)
- Async SQLite logging (non-blocking background thread)
- Flask-compatible streaming via SSE + regular JSON endpoints

PIPELINE:
  aquery() / aquery_stream()
    ├── analyze()          LLM call #1  (~0.4s with tight prompt)
    ├── route()            rule-based   (0 ms)
    ├── retrieve()         async/parallel across layers
    ├── evaluate()         heuristic    (0 ms)
    ├── generate/stream()  LLM call #2  (streamed → first token <1s)
    └── self_correct()     heuristic    (0 ms)
"""
import os
os.environ["HF_HOME"] = "/home/ubuntu/.cache/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
import re
import asyncio
import threading
import queue as stdlib_queue
import sqlite3
import time
import json
from typing import Dict, List, Tuple, Optional, Any, AsyncGenerator, Generator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import os
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

HF_CACHE = "/home/ubuntu/.cache/huggingface"
for _k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
           "HUGGINGFACE_HUB_CACHE", "HF_HUB_DISABLE_SYMLINKS_WARNING"):
    os.environ[_k] = HF_CACHE if _k != "HF_HUB_DISABLE_SYMLINKS_WARNING" else "1"

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TextIteratorStreamer,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==================== DOCUMENT LINK MAPPING ====================

DOCUMENT_LINKS: Dict[str, str] = {
    "Policy C-901":           "https://drive.google.com/mock-abbreviation-link",
    "Employee Handbook 2025": "https://drive.google.com/mock-handbook-link",
    "PTO Policy":             "https://drive.google.com/mock-pto-link",
}

# ==================== LOW CONFIDENCE CONTACT ====================

LOW_CONFIDENCE_THRESHOLD = 0.60

CONTACT_PERSON = {
    "name":  "Greg Smith",
    "email": "greg@company.com",
    "phone": "+1-555-123-4567",
}

# ==================== CONFIGURATION ====================

VECTOR_DB_DIR  = "/home/ubuntu/projects/RAG/chroma_db_fast/chroma_db_fast"
LLM_MODEL      = "/home/ubuntu/projects/risk_hospitalisation/models/Qwen2.5-3B-Instruct"
MONITORING_DB  = "./monitoring.db"

PRIORITY_ORDER       = ["company_policy", "medicare", "medical", "general"]
TOP_K                = 3
FETCH_K              = 15
MAX_NEW_TOKENS       = 180      # tighter cap → faster generation
CONFIDENCE_THRESHOLD = 0.65
ENABLE_MEMORY        = True
ENABLE_SELF_CORRECTION = True

# Hard ceiling on sentences in any final answer
MAX_ANSWER_SENTENCES = 4

# ==================== DATA STRUCTURES ====================

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
    query_type:        QueryType
    complexity:        float
    entities:          List[str]
    keywords:          List[str]
    requires_multi_step: bool
    layer_hints:       List[str]
    confidence:        float

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
    answer:            str
    citations:         str
    confidence:        float
    query_analysis:    QueryAnalysis
    retrieval_plan:    RetrievalPlan
    quality_assessment: QualityAssessment
    iterations:        int
    execution_time:    float
    agent_trace:       List[Dict[str, Any]]

@dataclass
class ConversationMemory:
    history:         List[Dict[str, str]] = field(default_factory=list)
    context_summary: str = ""
    max_history:     int = 5

# ==================== CONCISE TEXT UTILITIES ====================

# Compiled once at import time
_CITATION_PAT    = re.compile(r'\([^)]*?p\.\d+\)')
_CITE_ONLY_LINE  = re.compile(r'^\s*\([^)]*?p\.\d+\)\s*$')
_MULTI_SPACE     = re.compile(r'\s+')
_DOUBLE_DOT      = re.compile(r'\.\s*\.')
_TRAILING_CITES  = re.compile(r'(\([^)]*?p\.\d+\)\s*){3,}$')
_FILLER_STARTS   = re.compile(
    r'^(Note:|Please note|As mentioned|As stated|In summary|To summarize|'
    r'Based on the|According to the above|The above|This means that)', re.I
)

CUTOFF_MARKERS = [
    "Note:", "(Note:", "Please disregard", "I am following",
    "Sources:", "Citation:", "Reference:", "In summary,", "To summarize,",
]

SPECULATION_PHRASES = ["I believe", "I think", "I'm not sure", "possibly", "maybe"]


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]


def enforce_sentence_budget(text: str, budget: int = MAX_ANSWER_SENTENCES) -> str:
    """Keep only `budget` unique, non-filler sentences."""
    sentences = _split_sentences(text)
    seen: set = set()
    kept: List[str] = []
    for sent in sentences:
        norm = _CITATION_PAT.sub("", sent).strip().rstrip(".")
        if not norm or _FILLER_STARTS.match(norm):
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(sent)
        if len(kept) >= budget:
            break
    return " ".join(kept)


def clean_answer(answer: str) -> str:
    """
    Multi-pass cleaning pipeline.
    0. Strip any echoed system/prompt preamble before the answer marker
    1. Cut at hard stop markers
    2. Strip trailing citation clusters
    3. Remove citation-only lines
    4. Deduplicate inline citations (keep ≤2 each)
    5. Sentence budget + deduplication
    6. Normalise whitespace
    """
    # Pass 0 — strip echoed prompt up to the answer marker
    _MARKER = "Answer (2-3 sentences, no repetition):"
    if _MARKER in answer:
        answer = answer.split(_MARKER, 1)[-1]

    # Also catch the SYSTEM line if it leaked through
    _SYSTEM_LEAD = "You are a precise document Q&A assistant."
    if answer.strip().startswith(_SYSTEM_LEAD):
        # Find the first sentence that looks like an actual answer (not a rule)
        # by skipping lines that start with known instruction patterns
        _RULE_PAT = re.compile(r'^(You are|STRICT RULES|Rules:|Context:|Question:|Answer\s*\()', re.I)
        lines = answer.split("\n")
        real_lines = [l for l in lines if l.strip() and not _RULE_PAT.match(l.strip())]
        answer = " ".join(real_lines)

    # 1 — stop markers
    for marker in CUTOFF_MARKERS:
        if marker in answer:
            answer = answer.split(marker)[0]

    # 2 — trailing clusters
    answer = _TRAILING_CITES.sub("", answer)

    # 3 — citation-only lines
    lines = [
        ln.strip() for ln in answer.split("\n")
        if ln.strip() and not _CITE_ONLY_LINE.match(ln)
    ]
    answer = " ".join(lines)

    # 4 — deduplicate inline citations
    seen_cites: Dict[str, int] = {}

    def _repl(m: re.Match) -> str:
        c = m.group(0)
        seen_cites[c] = seen_cites.get(c, 0) + 1
        return c if seen_cites[c] <= 2 else ""

    answer = _CITATION_PAT.sub(_repl, answer)

    # 5 — sentence budget
    answer = enforce_sentence_budget(answer)

    # 6 — whitespace
    answer = _MULTI_SPACE.sub(" ", answer)
    answer = _DOUBLE_DOT.sub(".", answer)

    return answer.strip() or "Answer not found in available documents."


# ==================== STREAMING LLM ENGINE ====================

class LLMEngine:
    """
    Wraps HuggingFace model with:
      • generate()         — blocking, returns full string
      • generate_stream()  — sync generator, yields tokens
      • agenerate()        — async wrapper of generate()
      • agenerate_stream() — async generator, yields tokens
    A threading.Lock prevents concurrent GPU access.
    """

    def __init__(self, model_path: str):
            print("[LLM] Loading model …")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # ── Auto-detect if model is already AWQ-quantized ──────────────
            # AWQ models have quantization baked into weights — passing a
            # BitsAndBytesConfig on top causes a conflict and crashes.
            # We detect AWQ by reading config.json OR by the model path name.
            import json as _json

            _is_awq = False
            _cfg_file = os.path.join(model_path, "config.json")

            if os.path.exists(_cfg_file):
                try:
                    with open(_cfg_file) as f:
                        _cfg = _json.load(f)
                    _qt = _cfg.get("quantization_config", {})
                    _is_awq = (
                        str(_qt.get("quant_type",  "")).lower() == "awq"
                        or "awq" in str(_qt.get("backend", "")).lower()
                        or _cfg.get("model_type", "") == "awq"
                    )
                except Exception:
                    pass

            # Fallback: check model path name
            if not _is_awq:
                _is_awq = "awq" in model_path.lower()

            # ── Load model ─────────────────────────────────────────────────
            if _is_awq:
                print("[LLM] AWQ model detected — skipping BitsAndBytes config")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            else:
                print("[LLM] Loading with 4-bit BitsAndBytes quantization")
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quant,
                    device_map="auto",
                )

            self._lock = threading.Lock()
            print("[LLM] ✓ Model loaded")
    # ── internal ──────────────────────────────────────────────────────────

    def _encode(self, prompt: str) -> dict:
        return self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

    # Marker placed at the very end of every generation prompt.
    # We split on it to reliably strip the echoed prompt regardless of
    # minor whitespace differences introduced by tokenization/decoding.
    ANSWER_MARKER = "Answer (2-3 sentences, no repetition):"

    def _trim(self, full: str, prompt: str) -> str:
        """
        Robust prompt-stripping strategy (tried in order):
        1. Split on the known answer marker — works even when decoded text
           doesn't exactly match `prompt` due to tokenizer whitespace changes.
        2. Exact prefix strip.
        3. Fallback: return as-is (already stripped by skip_prompt=True in streamer).
        """
        if self.ANSWER_MARKER in full:
            return full.split(self.ANSWER_MARKER, 1)[-1].strip()
        if full.startswith(prompt):
            return full[len(prompt):].strip()
        return full.strip()

    # ── sync ──────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
        with self._lock:
            inputs = self._encode(prompt)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=None, do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._trim(text, prompt)

    def generate_stream(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> Generator[str, None, None]:
        """Streams tokens via TextIteratorStreamer on a background thread."""
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inputs = self._encode(prompt)

        def _run():
            with self._lock:
                with torch.no_grad():
                    self.model.generate(
                        **inputs, max_new_tokens=max_tokens,
                        temperature=None, do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                    )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for tok in streamer:
            yield tok
        t.join()

    # ── async ──────────────────────────────────────────────────────────────

    async def agenerate(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, max_tokens)

    async def agenerate_stream(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        tok_q: asyncio.Queue = asyncio.Queue()

        def _produce():
            for tok in self.generate_stream(prompt, max_tokens):
                loop.call_soon_threadsafe(tok_q.put_nowait, tok)
            loop.call_soon_threadsafe(tok_q.put_nowait, None)

        threading.Thread(target=_produce, daemon=True).start()

        while True:
            tok = await tok_q.get()
            if tok is None:
                break
            yield tok


# ==================== VECTOR DB ENGINE ====================

class VectorDBEngine:
    def __init__(self, db_dir: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
        )
        self.db = Chroma(persist_directory=db_dir, embedding_function=emb)
        print(f"[VectorDB] ✓ Connected to {db_dir}")

    def _search_sync(self, query: str, layer: str, k: int, use_mmr: bool) -> List[Document]:
        try:
            hits = (
                self.db.max_marginal_relevance_search(query, k=k * 3, fetch_k=FETCH_K)
                if use_mmr
                else self.db.similarity_search(query, k=k * 3)
            )
            filtered = [d for d in hits if d.metadata.get("layer") == layer]
            return (filtered or hits)[:k]
        except Exception as e:
            print(f"[VectorDB] WARN: {e}")
            return []

    async def search_async(self, query: str, layer: str,
                           k: int = TOP_K, use_mmr: bool = True) -> List[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query, layer, k, use_mmr)

    async def search_parallel(self, query: str, layers: List[str],
                               k_per_layer: int = 2) -> List[Document]:
        """Fetch ALL layers concurrently — key latency win."""
        results = await asyncio.gather(
            *[self.search_async(query, layer, k=k_per_layer) for layer in layers]
        )
        combined = [doc for chunk in results for doc in chunk]
        return combined[:10]


# ==================== AGENTS ====================

class QueryAnalyzerAgent:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    async def analyze(self, query: str, memory: Optional[ConversationMemory] = None) -> QueryAnalysis:
        ctx = (f"\nContext: {memory.context_summary[:100]}"
               if memory and memory.context_summary else "")
        prompt = (
            "Classify this query. Output ONLY 7 lines, no explanations:\n\n"
            f"Query: {query}{ctx}\n\n"
            "Example output:\n"
            "QUERY_TYPE: simple_fact\n"
            "COMPLEXITY: 0.6\n"
            "ENTITIES: vacation, policy\n"
            "KEYWORDS: vacation, days, policy\n"
            "MULTI_STEP: no\n"
            "LAYER_HINTS: company_policy\n"
            "CONFIDENCE: 0.8\n\n"
            "Output for the query above (7 lines only):\n"
            "QUERY_TYPE:"
        )
        raw = await self.llm.agenerate(prompt, max_tokens=100)
        # Re-attach the cut prefix so _parse sees "QUERY_TYPE: ..."
        raw = "QUERY_TYPE:" + raw
        return self._parse(raw)

    def _parse(self, raw: str) -> QueryAnalysis:
        data: Dict[str, str] = {}
        for line in raw.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip().upper()] = v.strip()

        qt = QueryType.POLICY_LOOKUP
        for q in QueryType:
            if q.value in data.get("QUERY_TYPE", "").lower():
                qt = q
                break

        def _f(key: str, dflt: float) -> float:
            try:
                nums = [x for x in data.get(key, str(dflt)).split()
                        if x.replace(".", "", 1).isdigit()]
                return max(0.0, min(1.0, float(nums[0]) if nums else dflt))
            except Exception:
                return dflt

        lh = [l.strip() for l in data.get("LAYER_HINTS", "company_policy").split(",")
              if l.strip() in PRIORITY_ORDER]
        return QueryAnalysis(
            query_type=qt,
            complexity=_f("COMPLEXITY", 0.6),
            entities=[e.strip() for e in data.get("ENTITIES", "").split(",") if e.strip()][:5],
            keywords=[k.strip() for k in data.get("KEYWORDS", "").split(",") if k.strip()][:5],
            requires_multi_step="yes" in data.get("MULTI_STEP", "no").lower(),
            layer_hints=lh or ["company_policy", "general"],
            confidence=_f("CONFIDENCE", 0.7),
        )


class RouterAgent:
    """Zero-latency rule-based routing — no LLM call."""

    def route(self, analysis: QueryAnalysis) -> RetrievalPlan:
        if analysis.query_type == QueryType.SIMPLE_FACT and analysis.complexity < 0.4:
            return RetrievalPlan(
                strategy=RetrievalStrategy.SINGLE_LAYER,
                layers=analysis.layer_hints[:1],
                num_chunks=4, use_mmr=True,
                refinement_needed=False, reasoning="Simple fact",
            )
        if analysis.query_type in (QueryType.COMPARATIVE, QueryType.MULTI_STEP):
            return RetrievalPlan(
                strategy=RetrievalStrategy.MULTI_LAYER,
                layers=analysis.layer_hints[:2],
                num_chunks=6, use_mmr=True,
                refinement_needed=False, reasoning="Complex query",
            )
        return RetrievalPlan(
            strategy=RetrievalStrategy.SINGLE_LAYER,
            layers=analysis.layer_hints[:1],
            num_chunks=5, use_mmr=True,
            refinement_needed=False, reasoning="Standard lookup",
        )


class RetrievalExecutor:
    def __init__(self, db: VectorDBEngine):
        self.db = db

    async def execute(self, query: str, plan: RetrievalPlan) -> List[Document]:
        if plan.strategy == RetrievalStrategy.MULTI_LAYER:
            return await self.db.search_parallel(query, plan.layers, k_per_layer=3)
        layer = plan.layers[0] if plan.layers else "company_policy"
        return await self.db.search_async(query, layer, k=plan.num_chunks)


class QualityEvaluatorAgent:
    """Instant heuristic — zero LLM calls."""

    STOP_WORDS = {
        "the","a","an","is","are","was","were","in","on","at","to","of",
        "and","or","for","with","that","this","it","be","as","by","from",
        "have","has","do","does",
    }

    def evaluate(self, query: str, documents: List[Document]) -> QualityAssessment:
        if not documents:
            return QualityAssessment(0, 0, 0, 0, True, ["No docs"], "No docs")

        qw = set(re.findall(r"\w+", query.lower()))
        dw = set(re.findall(r"\w+",
                            " ".join(d.page_content[:300] for d in documents[:4]).lower()))
        meaningful = qw - self.STOP_WORDS or qw
        relevance  = min(0.95, len(meaningful & dw) / max(len(meaningful), 1) * 1.5)

        n = len(documents)
        coverage = 0.90 if n >= 5 else (0.75 if n >= 3 else (0.55 if n >= 1 else 0.20))

        us = len({d.metadata.get("source", "") for d in documents})
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
    """
    Strict prompt enforces conciseness at the model level.
    Supports both async full-string and async streaming.
    """

    SYSTEM = (
        "You are a precise document Q&A assistant. "
        "STRICT RULES: (1) Answer in 2-3 sentences MAXIMUM. "
        "(2) State each fact ONCE — never repeat. "
        "(3) Cite source inline like (Handbook p.12). "
        "(4) No notes, disclaimers, or summaries. "
        "(5) Stop immediately after the answer."
    )

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def _context(self, docs: List[Document]) -> str:
        seen, blocks = set(), []
        for doc in docs[:4]:
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            key  = (src, page)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(f"[{src} p.{page}]\n{doc.page_content.strip()[:600]}")
        return "\n\n".join(blocks)

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

    # Must match LLMEngine.ANSWER_MARKER exactly so _trim() can split on it
    _MARKER = "Answer (2-3 sentences, no repetition):"

    def _prompt(self, query: str, docs: List[Document]) -> str:
        return (
            f"{self.SYSTEM}\n\n"
            f"Context:\n{self._context(docs)}\n\n"
            f"Question: {query}\n\n"
            f"{self._MARKER}"
        )

    async def agenerate(self, query: str, docs: List[Document],
                        quality: QualityAssessment) -> Tuple[str, str, float]:
        raw    = await self.llm.agenerate(self._prompt(query, docs), MAX_NEW_TOKENS)
        answer = clean_answer(raw)
        return answer, self._citations(docs), quality.overall_quality

    async def agenerate_stream(self, query: str,
                               docs: List[Document]) -> AsyncGenerator[str, None]:
        async for tok in self.llm.agenerate_stream(self._prompt(query, docs), MAX_NEW_TOKENS):
            yield tok

    # sync versions kept for CLI fallback
    def generate(self, query: str, docs: List[Document],
                 quality: QualityAssessment) -> Tuple[str, str, float]:
        raw    = self.llm.generate(self._prompt(query, docs), MAX_NEW_TOKENS)
        answer = clean_answer(raw)
        return answer, self._citations(docs), quality.overall_quality


class SelfCorrectionAgent:
    """Heuristic penalties — zero LLM calls, instant."""

    def correct(self, answer: str, base: float) -> Tuple[str, float, List[str]]:
        issues, conf = [], base

        if answer.count("Note:") > 1:
            issues.append("repetitive_notes")
            conf -= 0.10
            answer = answer.split("Note:")[0].strip()

        for p in SPECULATION_PHRASES:
            if p.lower() in answer.lower():
                issues.append("speculation")
                conf -= 0.08
                break

        if len(answer.strip()) < 40:
            issues.append("too_short")
            conf -= 0.12

        if "answer not found" in answer.lower():
            issues.append("no_info")
            conf -= 0.20

        return answer, round(max(0.0, min(1.0, conf)), 4), issues


class MemoryManager:
    def __init__(self):
        self._mem = ConversationMemory()

    def add(self, query: str, answer: str):
        self._mem.history.append({"query": query, "answer": (answer or "")[:200]})
        self._mem.history = self._mem.history[-self._mem.max_history:]
        recent = self._mem.history[-2:]
        self._mem.context_summary = " | ".join(f"Q:{t['query'][:50]}" for t in recent)

    def get(self) -> ConversationMemory:
        return self._mem

    def reset(self):
        self._mem = ConversationMemory()


# ==================== NON-BLOCKING MONITOR ====================

class SimpleMonitor:
    """Background-thread writer — DB writes never stall the request path."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._q: stdlib_queue.Queue = stdlib_queue.Queue()
        conn = sqlite3.connect(db_path)
        # Use IF NOT EXISTS — if the table already exists with extra columns
        # from a previous version, drop it so we start clean.
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
                print(f"[Monitor] {e}")

    def log(self, query: str, answer: str, confidence: float,
            exec_time: float, iterations: int, strategy: str, success: bool):
        self._q.put((
            datetime.now().isoformat(), query, answer[:500],
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


# ==================== CONTACT HELPER ====================

def contact_block(c: Dict[str, str]) -> str:
    return (
        "\n\n⚠️  Low confidence — answer may be incomplete.\n"
        f"Contact: {c['name']} | {c['email']} | {c['phone']}"
    )


# ==================== ORCHESTRATOR ====================

class AgenticRAGOrchestrator:
    """
    All public APIs are async-first.
    Sync wrappers provided for CLI and plain Flask use.
    """

    def __init__(self, llm_path: str, vectordb_path: str, monitoring_db: str):
        print("\n🤖 Initializing Agentic RAG v3 …")
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
        print("✅ System ready.\n")

    # ── async full-response ───────────────────────────────────────────────

    async def aquery(
        self,
        question: str,
        enable_memory: bool = ENABLE_MEMORY,
        enable_self_correction: bool = ENABLE_SELF_CORRECTION,
    ) -> AgenticResponse:
        t0  = time.perf_counter()
        mem = self.memory.get() if enable_memory else None

        print("🔍 Analyzing …")
        analysis = await self.analyzer.analyze(question, mem)
        print(f"   type={analysis.query_type.value}  complexity={analysis.complexity:.2f}")

        plan = self.router.route(analysis)
        print(f"   strategy={plan.strategy.value}  layers={plan.layers}")

        print("📚 Retrieving …")
        docs = await self.retriever.execute(question, plan)
        print(f"   {len(docs)} docs")

        if not docs:
            return self._empty(question, analysis, plan, time.perf_counter() - t0)

        quality = self.evaluator.evaluate(question, docs)
        print(f"   quality={quality.overall_quality:.2f}")

        print("💬 Generating …")
        answer, citations, confidence = await self.generator.agenerate(question, docs, quality)

        if enable_self_correction:
            answer, confidence, issues = self.corrector.correct(answer, confidence)
            if issues:
                print(f"   ⚠ fixed: {', '.join(issues)}")

        exec_time    = round(time.perf_counter() - t0, 3)
        final_answer = self._assemble(answer, citations, confidence)

        if enable_memory:
            self.memory.add(question, answer)
        self.monitor.log(question, answer, confidence, exec_time, 1, plan.strategy.value, True)
        print(f"✅ {exec_time}s | conf={confidence:.0%}")

        return AgenticResponse(
            answer=final_answer, citations=citations, confidence=confidence,
            query_analysis=analysis, retrieval_plan=plan,
            quality_assessment=quality, iterations=1,
            execution_time=exec_time, agent_trace=[],
        )

    # ── async streaming ───────────────────────────────────────────────────

    async def aquery_stream(
        self,
        question: str,
        enable_memory: bool = ENABLE_MEMORY,
    ) -> AsyncGenerator[str, None]:
        """
        Yields newline-delimited JSON strings (SSE-ready):
          {"type":"token",    "data":"..."}
          {"type":"citations","data":"..."}
          {"type":"meta",     "data":{...}}
          {"type":"error",    "data":"..."}
        """
        t0  = time.perf_counter()
        mem = self.memory.get() if enable_memory else None

        try:
            analysis = await self.analyzer.analyze(question, mem)
            plan     = self.router.route(analysis)
            docs     = await self.retriever.execute(question, plan)

            if not docs:
                yield json.dumps({"type": "token",
                                  "data": "No relevant documents found."})
                yield json.dumps({"type": "token",
                                  "data": contact_block(CONTACT_PERSON)})
                yield json.dumps({"type": "meta",
                                  "data": {"confidence": 0.0,
                                           "execution_time": round(time.perf_counter()-t0, 3)}})
                return

            quality    = self.evaluator.evaluate(question, docs)
            citations  = self.generator._citations(docs)
            buffer     = ""

            async for tok in self.generator.agenerate_stream(question, docs):
                buffer += tok
                yield json.dumps({"type": "token", "data": tok})

            # Post-process assembled buffer
            clean      = clean_answer(buffer)
            confidence = quality.overall_quality
            clean, confidence, _ = self.corrector.correct(clean, confidence)
            exec_time  = round(time.perf_counter() - t0, 3)

            yield json.dumps({"type": "citations", "data": citations})
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                yield json.dumps({"type": "token", "data": contact_block(CONTACT_PERSON)})
            yield json.dumps({"type": "meta", "data": {
                "confidence":    confidence,
                "execution_time": exec_time,
                "quality":       quality.feedback,
            }})

            if enable_memory:
                self.memory.add(question, clean)
            self.monitor.log(question, clean, confidence, exec_time, 1, plan.strategy.value, True)

        except Exception as exc:
            yield json.dumps({"type": "error", "data": str(exc)})

    # ── sync wrappers ─────────────────────────────────────────────────────

    def query_sync(self, question: str, **kw) -> AgenticResponse:
        """Blocking — runs async pipeline in a fresh event loop."""
        return asyncio.run(self.aquery(question, **kw))

    def stream_sync(self, question: str, **kw) -> Generator[str, None, None]:
        """
        Sync generator for Flask streaming responses.
        Bridges the async generator to a regular generator.
        Each yielded item is a newline-terminated JSON string.
        """
        q: stdlib_queue.Queue = stdlib_queue.Queue()

        async def _collect():
            async for chunk in self.aquery_stream(question, **kw):
                q.put(chunk)
            q.put(None)

        threading.Thread(target=lambda: asyncio.run(_collect()), daemon=True).start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item + "\n"

    # ── helpers ───────────────────────────────────────────────────────────

    def _assemble(self, answer: str, citations: str, confidence: float) -> str:
        parts = [answer, f"\nSources:\n{citations}"]
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            parts.append(contact_block(CONTACT_PERSON))
        return "".join(parts)

    def _empty(self, question, analysis, plan, t) -> AgenticResponse:
        msg = "No relevant documents found." + contact_block(CONTACT_PERSON)
        return AgenticResponse(
            answer=msg, citations="", confidence=0.0,
            query_analysis=analysis, retrieval_plan=plan,
            quality_assessment=QualityAssessment(0,0,0,0,False,[],""),
            iterations=1, execution_time=round(t,3), agent_trace=[],
        )

    def reset_memory(self):
        self.memory.reset()
        print("🔄 Memory reset")

    def show_stats(self):
        s = self.monitor.get_stats()
        if s:
            print(f"\n📊 {s['total_queries']} queries | "
                  f"avg {s['avg_time']}s | conf {s['avg_confidence']:.0%} | "
                  f"success {s['success_rate']}%")
        else:
            print("No stats yet.")


# ==================== CLI ====================

def main():
    print("\n" + "="*60)
    print("🤖 AGENTIC RAG v3 — Async + Streaming + Concise")
    print("="*60)
    print("Commands: <question>  |  stream <question>  |  reset  |  stats  |  exit\n")

    orch = AgenticRAGOrchestrator(LLM_MODEL, VECTOR_DB_DIR, MONITORING_DB)

    while True:
        try:
            raw = input("\n💬 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            print("👋 Bye!")
            break
        if raw.lower() == "reset":
            orch.reset_memory()
            continue
        if raw.lower() == "stats":
            orch.show_stats()
            continue

        if raw.lower().startswith("stream "):
            question = raw[7:].strip()
            print("\n✨ STREAMING\n" + "-"*60)
            for chunk in orch.stream_sync(question):
                try:
                    obj = json.loads(chunk.strip())
                    if obj["type"] == "token":
                        print(obj["data"], end="", flush=True)
                    elif obj["type"] == "citations":
                        print(f"\n\nSources:\n{obj['data']}")
                    elif obj["type"] == "meta":
                        m = obj["data"]
                        print(f"\n[conf={m['confidence']:.0%}  {m['execution_time']}s]")
                    elif obj["type"] == "error":
                        print(f"\n❌ {obj['data']}")
                except Exception:
                    print(chunk, end="")
            print("\n" + "="*60)
            continue

        try:
            resp = orch.query_sync(raw)
            print("\n" + "="*60)
            print(resp.answer)
            print(f"\n📊 conf={resp.confidence:.0%}  time={resp.execution_time}s")
            print("="*60)
        except Exception as e:
            print(f"\n❌ {e}")


if __name__ == "__main__":
    main()


