# stdlib
import os
import warnings
import json
import re
import time
import uuid
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from typing import Optional

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# HuggingFace cache paths — must be set before importing transformers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = "/home/ubuntu/projects/RAG" if os.path.exists("/home/ubuntu/projects/RAG") else BASE_DIR
HF_CACHE = os.path.join(PROJECT_ROOT, "cache", "huggingface")
for _k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
           "HUGGINGFACE_HUB_CACHE"):
    os.environ[_k] = HF_CACHE
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs(HF_CACHE, exist_ok=True)

# third party
from flask import Flask, request, jsonify, Response, g, stream_with_context

# local
from agentic_rag import (
    AgenticRAGOrchestrator,
    ConversationMemory,
    LLM_MODEL,
    VECTOR_DB_DIR,
    MONITORING_DB,
    ENABLE_MEMORY,
    ENABLE_SELF_CORRECTION,
    clean_answer,
    _word_set,
    _split_sentences,
    CONTACT_DEPARTMENTS,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RATE_LIMIT_MAX_REQUESTS = 10        # requests per window per session
RATE_LIMIT_WINDOW_SECS  = 60        # rolling window
SSE_HEARTBEAT_SECS      = 15        # keep-alive comment interval
MAX_QUESTION_LENGTH     = 1_000     # characters
ALLOWED_ORIGINS         = ["*"]     # set to specific domains in production
SESSION_HEADER          = "X-Session-Id"

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rag_api")

# ══════════════════════════════════════════════════════════════════════════════
# ANSWER SANITISER
# ══════════════════════════════════════════════════════════════════════════════

# Characters/sequences produced by format_final_response that must never leak
# into the API answer field.
_BOX_CHARS   = re.compile(r'[─═]+')
_SECTION_HDR = re.compile(
    r'\s*(Answer|📄\s*Sources|📊\s*Confidence.*|⏱.*)\s*', re.I
)
_MULTI_SPACE = re.compile(r'\s{2,}')
_MULTI_NL    = re.compile(r'\n{2,}')

# Repeated-sentence detector (catches verbatim or near-verbatim duplication)
def _dedup_sentences(text: str) -> str:
    """Remove consecutive near-duplicate sentences from the answer."""
    sentences = _split_sentences(text.strip())
    seen: list[str] = []
    for s in sentences:
        # Normalise for comparison: lowercase, strip punctuation
        key = re.sub(r'[^\w\s]', '', s.lower()).strip()
        if not key:
            continue
        # Drop if very similar to the previous kept sentence
        if seen:
            prev_key = re.sub(r'[^\w\s]', '', seen[-1].lower()).strip()
            prev_words = set(prev_key.split())
            curr_words = set(key.split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))
                if overlap > 0.70:        # 70 % word-overlap → duplicate
                    log.debug("[Dedup] Dropped near-duplicate sentence: %s", s[:60])
                    continue
        seen.append(s)
    return " ".join(seen)


def sanitise_answer(raw_answer: str) -> str:
    """
    Strip all CLI-formatting artefacts from an answer string so it is safe
    to embed directly in a JSON response body.

    Handles the output of format_final_response (box-drawing chars, section
    headers, confidence/timing lines, stray newlines) as well as repeated
    sentences produced by the LLM.
    """
    text = raw_answer

    # 1. If the full format_final_response block was stored, extract only the
    #    text between the "Answer" header and the "📄 Sources" / "📊" lines.
    if "Answer" in text and ("📄" in text or "📊" in text):
        # Grab everything after the first "Answer" line
        after_header = re.split(r'Answer\s*\n', text, maxsplit=1)
        if len(after_header) > 1:
            text = after_header[1]
        # Drop everything from "📄 Sources" or "📊 Confidence" onwards
        text = re.split(r'📄\s*Sources|📊\s*Confidence', text)[0]

    # 2. Strip box-drawing lines (─────, ═════, etc.)
    text = _BOX_CHARS.sub('', text)

    # 3. Strip leftover section-header words
    text = _SECTION_HDR.sub(' ', text)

    # 4. Collapse whitespace
    text = _MULTI_NL.sub(' ', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = text.strip()

    # 5. Remove duplicate/redundant sentences
    text = _dedup_sentences(text)

    # 6. Final guard: if result is empty or only punctuation, return a clean fallback
    if not text or not re.search(r'[a-zA-Z]{3,}', text):
        return "I'm sorry, I was unable to find that information in the available documents."

    return text


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── startup state ─────────────────────────────────────────────────────────────
_ready        = False          # becomes True once model is loaded
_startup_time = time.time()
_orch: Optional[AgenticRAGOrchestrator] = None

# ── per-session memory store  { session_id → ConversationMemory } ──────────
_session_memories: dict = {}
_session_lock = threading.Lock()

# ── per-session rate-limit store  { session_id → [timestamps] } ───────────
_rate_store: dict = defaultdict(list)
_rate_lock  = threading.Lock()


def _load_model():
    """Runs in a background thread so Flask can start accepting /health immediately."""
    global _orch, _ready
    log.info("Loading RAG system …")
    _orch  = AgenticRAGOrchestrator(LLM_MODEL, VECTOR_DB_DIR, MONITORING_DB)
    _ready = True
    log.info("RAG system ready ✅")


threading.Thread(target=_load_model, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — session & rate-limiting
# ══════════════════════════════════════════════════════════════════════════════

def _get_session_id() -> str:
    """Return session ID from header, or create a new one."""
    sid = request.headers.get(SESSION_HEADER, "").strip()
    return sid if sid else str(uuid.uuid4())


def _get_or_create_memory(session_id: str) -> ConversationMemory:
    with _session_lock:
        if session_id not in _session_memories:
            _session_memories[session_id] = ConversationMemory()
        return _session_memories[session_id]


def _check_rate_limit(session_id: str) -> bool:
    """Returns True if the request is allowed, False if rate-limited."""
    now = time.time()
    with _rate_lock:
        timestamps = _rate_store[session_id]
        _rate_store[session_id] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECS]
        if len(_rate_store[session_id]) >= RATE_LIMIT_MAX_REQUESTS:
            return False
        _rate_store[session_id].append(now)
        return True

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — serialisation
# ══════════════════════════════════════════════════════════════════════════════

def _extract_clean_answer(response) -> str:
    """
    Pull a clean plain-text answer out of an AgenticResponse, regardless of
    whether the orchestrator stored a formatted or unformatted string.
    """
    raw = response.answer or ""
    clean = sanitise_answer(raw)
    is_fallback = "unable to find that information" in clean.lower() or "unable to find relevant information" in clean.lower()
    if response.confidence < 0.55 or is_fallback:
        parts = []
        for dept, names in CONTACT_DEPARTMENTS.items():
            parts.append(f"{dept}: {names}")
        contact_suffix = " For further assistance, please contact: " + " | ".join(parts) + "."
        if "Human Resources: Greg" not in clean:
            clean = clean.strip() + " " + contact_suffix
    return clean


def _serialize(response, clean_ans: str) -> dict:
    """Convert AgenticResponse dataclass → JSON-serialisable dict.

    Uses pre-sanitised `clean_ans` for the answer field so the caller
    controls what reaches the wire.
    """
    return {
        "answer":         clean_ans,
        "citations":      _clean_citations(response.citations),
        "confidence":     response.confidence,
        "execution_time": response.execution_time,
        "iterations":     response.iterations,
        "retrieved_chunks": getattr(response, "retrieved_chunks", []),
        "query_analysis": {
            "query_type":          response.query_analysis.query_type.value,
            "complexity":          response.query_analysis.complexity,
            "entities":            response.query_analysis.entities,
            "keywords":            response.query_analysis.keywords,
            "requires_multi_step": response.query_analysis.requires_multi_step,
            "layer_hints":         response.query_analysis.layer_hints,
        },
        "retrieval_plan": {
            "strategy":   response.retrieval_plan.strategy.value,
            "layers":     response.retrieval_plan.layers,
            "num_chunks": response.retrieval_plan.num_chunks,
            "reasoning":  response.retrieval_plan.reasoning,
        },
        "quality_assessment": {
            "relevance_score":   response.quality_assessment.relevance_score,
            "coverage_score":    response.quality_assessment.coverage_score,
            "consistency_score": response.quality_assessment.consistency_score,
            "overall_quality":   response.quality_assessment.overall_quality,
            "needs_refinement":  response.quality_assessment.needs_refinement,
            "missing_aspects":   response.quality_assessment.missing_aspects,
            "feedback":          response.quality_assessment.feedback,
        },
    }


def _clean_citations(raw_citations: str) -> list[dict]:
    """
    Convert the raw newline-delimited citations string into a clean list of
    structured objects, e.g.:

        [
            {"source": "Medicare Policies", "page": "3", "link": "https://..."},
            ...
        ]

    This prevents the citations blob from being a raw multi-line string in JSON.
    """
    if not raw_citations:
        return []

    results = []
    seen    = set()
    for line in raw_citations.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Expected format: "Filename (p.N) – URL"
        m = re.match(r'^(.+?)\s*\(p\.([^)]+)\)\s*[–-]\s*(.*)$', line)
        if m:
            source = m.group(1).strip()
            page   = m.group(2).strip()
            link   = m.group(3).strip()
        else:
            # Fallback: store line as-is under "source"
            source = line
            page   = ""
            link   = ""

        key = (source, page)
        if key in seen:
            continue
        seen.add(key)

        entry = {"source": source, "page": page}
        if link and link.lower() not in ("link not available", ""):
            entry["link"] = link
        results.append(entry)

    return results


def _error(message: str, code: int, detail: str = "") -> tuple:
    body = {"success": False, "error": message}
    if detail:
        body["detail"] = detail
    return jsonify(body), code


def _sse(obj: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(obj)}\n\n"

# ══════════════════════════════════════════════════════════════════════════════
# DECORATORS
# ══════════════════════════════════════════════════════════════════════════════

def require_ready(f):
    """Return 503 if the model hasn't finished loading."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _ready:
            return _error("Service starting up — please retry in a moment.", 503)
        return f(*args, **kwargs)
    return wrapper


def with_session(f):
    """
    Inject session_id into Flask's g object.
    Also enforces rate limiting and adds session_id to every response header.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.session_id = _get_session_id()

        if not _check_rate_limit(g.session_id):
            resp, code = _error(
                "Rate limit exceeded.",
                429,
                f"Max {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECS}s per session.",
            )
            resp.headers[SESSION_HEADER] = g.session_id
            return resp, code

        result = f(*args, **kwargs)

        # Attach session header to response
        if isinstance(result, tuple):
            resp, code = result
            resp.headers[SESSION_HEADER] = g.session_id
            return resp, code

        if isinstance(result, Response):
            result.headers[SESSION_HEADER] = g.session_id
        return result
    return wrapper


def log_request(f):
    """Log each request with timing."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0  = time.perf_counter()
        sid = getattr(g, "session_id", "—")
        log.info("→ %s %s  session=%s", request.method, request.path, sid)
        result = f(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info("← %s %s  %.2fs", request.method, request.path, elapsed)
        return result
    return wrapper

# ══════════════════════════════════════════════════════════════════════════════
# CORS  (minimal — no flask-cors dependency)
# ══════════════════════════════════════════════════════════════════════════════

@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    if "*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"]  = origin or "*"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, X-Session-Id, Authorization"
        )
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Expose-Headers"] = SESSION_HEADER
    return response


@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return "", 204

# ══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _validate_query_body(data: dict) -> Optional[str]:
    """Return an error string if invalid, else None."""
    question = (data.get("question") or "").strip()
    if not question:
        return "Field 'question' is required and must be a non-empty string."
    if len(question) > MAX_QUESTION_LENGTH:
        return f"Field 'question' must be ≤ {MAX_QUESTION_LENGTH} characters."
    if not isinstance(data.get("enable_memory", True), bool):
        return "Field 'enable_memory' must be a boolean."
    if not isinstance(data.get("enable_self_correction", True), bool):
        return "Field 'enable_self_correction' must be a boolean."
    return None

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── GET /health ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """
    Liveness + readiness probe.

    Response 200 (ready):
        {
            "status":  "ok",
            "ready":   true,
            "version": "v3",
            "uptime_seconds": 42.1
        }

    Response 503 (still loading):
        { "status": "starting", "ready": false }
    """
    uptime = round(time.time() - _startup_time, 1)
    if _ready:
        return jsonify({
            "status":         "ok",
            "ready":          True,
            "version":        "v3",
            "uptime_seconds": uptime,
        }), 200
    return jsonify({
        "status":         "starting",
        "ready":          False,
        "uptime_seconds": uptime,
    }), 503


# ── POST /query ───────────────────────────────────────────────────────────────

@app.route("/query", methods=["POST"])
@require_ready
@with_session
@log_request
def query():
    """
    Full blocking JSON answer.

    Request  (JSON):
    ┌────────────────────────────┬──────────┬───────────────────────────────┐
    │ Field                      │ Type     │ Description                   │
    ├────────────────────────────┼──────────┼───────────────────────────────┤
    │ question          (req)    │ string   │ The question to ask           │
    │ enable_memory     (opt)    │ bool     │ Use conversation memory       │
    │ enable_self_correction(opt)│ bool     │ Apply self-correction         │
    └────────────────────────────┴──────────┴───────────────────────────────┘

    Response 200:
    {
        "success":     true,
        "session_id":  "abc-123",
        "answer":      "Employees receive 15 PTO days per year (Handbook p.18).",
        "citations":   [{"source": "...", "page": "18", "link": "https://..."}],
        "confidence":  0.84,
        "execution_time": 3.2,
        "iterations":  1,
        "query_analysis":    { ... },
        "retrieval_plan":    { ... },
        "quality_assessment":{ ... }
    }

    Error responses:
        400 — validation error
        429 — rate limit exceeded
        500 — internal server error
        503 — model still loading
    """
    data = request.get_json(silent=True) or {}
    err  = _validate_query_body(data)
    if err:
        return _error(err, 400)

    question               = data["question"].strip()
    enable_memory          = data.get("enable_memory", ENABLE_MEMORY)
    enable_self_correction = data.get("enable_self_correction", ENABLE_SELF_CORRECTION)

    if enable_memory:
        _orch.memory._mem = _get_or_create_memory(g.session_id)

    try:
        resp      = _orch.query_sync(
            question,
            session_id=g.session_id,
            enable_memory=enable_memory,
            enable_self_correction=enable_self_correction,
        )
        clean_ans = _extract_clean_answer(resp)
        result    = _serialize(resp, clean_ans)
        result.update({"success": True, "session_id": g.session_id})
        return jsonify(result), 200

    except Exception as exc:
        log.exception("query() error")
        return _error("Internal server error.", 500, str(exc))


# ── POST /upload ──────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
@require_ready
@with_session
@log_request
def upload():
    """
    Upload a new PDF and update the RAG knowledge base.

    Request: multipart/form-data with field "file" (PDF).

    Response 200:
        {
            "success": true,
            "message": "File uploaded. Processing in background.",
            "session_id": "abc-123"
        }
    """
    if "file" not in request.files:
        return _error("No file provided.", 400)

    file = request.files["file"]
    if file.filename == "":
        return _error("Empty filename.", 400)

    tmp_dir = os.path.join(PROJECT_ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = os.path.join(tmp_dir, file.filename)
    file.save(save_path)

    try:
        def _run():
            try:
                _orch.ingest_document(save_path)
            except Exception as e:
                log.error("Ingestion failed: %s", e)

        threading.Thread(target=_run, daemon=True).start()

        return jsonify({
            "success":    True,
            "message":    "File uploaded. Processing in background.",
            "session_id": g.session_id,
        }), 200

    except Exception as exc:
        log.exception("upload() error")
        return _error("Upload failed.", 500, str(exc))


# ── POST /delete ──────────────────────────────────────────────────────────────

@app.route("/delete", methods=["POST"])
@require_ready
@with_session
@log_request
def delete_doc():
    """
    Delete a document from the RAG knowledge base.

    Request (JSON):
        {
            "filename": "Employee Handbook 2025 - Copy.pdf"
        }

    Response 200:
        {
            "success": true,
            "message": "Document 'filename.pdf' deleted from vector DB.",
            "session_id": "abc-123"
        }
    """
    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "").strip()
    if not filename:
        return _error("Field 'filename' is required.", 400)

    try:
        success = _orch.delete_document(filename)
        if success:
            return jsonify({
                "success":    True,
                "message":    f"Document '{filename}' deleted from vector DB.",
                "session_id": g.session_id,
            }), 200
        else:
            return _error(f"Failed to delete document '{filename}'.", 500)
    except Exception as exc:
        log.exception("delete_doc() error")
        return _error("Delete failed.", 500, str(exc))


# ── POST /query/stream ────────────────────────────────────────────────────────

@app.route("/query/stream", methods=["POST"])
@require_ready
@with_session
@log_request
def query_stream():
    """
    Server-Sent Events (SSE) streaming endpoint.
    First token typically arrives in < 1 second.

    Request  (JSON): same as POST /query (enable_self_correction not supported in stream mode)

    Response: Content-Type: text/event-stream
    ─────────────────────────────────────────────────────────────
    Each event is a newline-delimited JSON object on a `data:` line:

      data: {"type": "start",    "session_id": "abc-123"}

      data: {"type": "status",   "data": "Searching policies..."}
      data: {"type": "status",   "data": "Retrieving documents..."}

      data: {"type": "token",    "data": "According "}
      data: {"type": "token",    "data": "to the Handbook"}
      ...

      data: {"type": "citations","data": [{"source":"...","page":"18","link":"..."}]}

      data: {"type": "meta",     "data": {
                                    "confidence": 0.84,
                                    "execution_time": 3.1,
                                    "quality": "rel=0.90 cov=0.75 cons=0.80"
                                 }}

      data: {"type": "done"}

    On error:
      data: {"type": "error", "data": "<message>"}

    Heartbeat (every 15s while waiting):
      : heartbeat
    ─────────────────────────────────────────────────────────────
    """
    data = request.get_json(silent=True) or {}
    err  = _validate_query_body(data)
    if err:
        return _error(err, 400)

    question      = data["question"].strip()
    enable_memory = data.get("enable_memory", ENABLE_MEMORY)
    session_id    = g.session_id

    if enable_memory:
        _orch.memory._mem = _get_or_create_memory(session_id)

    # Progress messages produced by the orchestrator that should NOT be emitted
    # as answer tokens — they go out as "status" events instead.
    _STATUS_TOKENS = frozenset({
        "🔍 Searching policies…",
        "📚 Retrieving documents…",
        "🔍 Searching policies...",
        "📚 Retrieving documents...",
    })

    @stream_with_context
    def generate():
        yield _sse({"type": "start", "session_id": session_id})

        last_heartbeat = time.time()
        token_buffer   = ""           # accumulate tokens for post-processing

        try:
            for raw_chunk in _orch.stream_sync(question, session_id=session_id, enable_memory=enable_memory):
                # Heartbeat to keep the connection alive
                if time.time() - last_heartbeat > SSE_HEARTBEAT_SECS:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.time()

                raw_chunk = raw_chunk.strip()
                if not raw_chunk:
                    continue

                try:
                    obj = json.loads(raw_chunk)
                except json.JSONDecodeError:
                    # Shouldn't happen — yield as token anyway
                    yield _sse({"type": "token", "data": raw_chunk})
                    token_buffer += raw_chunk
                    continue

                event_type = obj.get("type", "")
                event_data = obj.get("data", "")

                if event_type == "token":
                    tok = event_data or ""

                    # Route internal progress messages as status events
                    if tok.strip() in _STATUS_TOKENS:
                        # Strip emoji/ellipsis for a cleaner status label
                        label = re.sub(r'[🔍📚…]', '', tok).strip().rstrip('…').strip()
                        yield _sse({"type": "status", "data": label})
                        continue

                    # Filter out box-drawing characters leaking into stream
                    if _BOX_CHARS.fullmatch(tok.strip()) or tok.strip() in ("Answer", ""):
                        continue

                    token_buffer += tok
                    yield _sse({"type": "token", "data": tok})

                elif event_type == "citations":
                    # Parse raw citation string into structured list
                    structured = _clean_citations(event_data if isinstance(event_data, str) else "")
                    yield _sse({"type": "citations", "data": structured})

                elif event_type == "meta":
                    # Pass meta through unchanged (confidence, timing, quality)
                    yield _sse({"type": "meta", "data": event_data})

                elif event_type == "error":
                    yield _sse({"type": "error", "data": event_data})

                else:
                    # Forward any unknown event types transparently
                    yield _sse(obj)

        except Exception as exc:
            log.exception("query_stream() error")
            yield _sse({"type": "error", "data": str(exc)})

        yield _sse({"type": "done"})

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",        # disable nginx buffering
            "Connection":        "keep-alive",
            SESSION_HEADER:      session_id,
        },
    )


# ── POST /reset ───────────────────────────────────────────────────────────────

@app.route("/reset", methods=["POST"])
@require_ready
@with_session
@log_request
def reset():
    """
    Reset conversation memory for the current session.

    No request body required.

    Response 200:
        {
            "success":    true,
            "session_id": "abc-123",
            "message":    "Memory cleared for session abc-123"
        }
    """
    try:
        with _session_lock:
            _session_memories.pop(g.session_id, None)
        _orch.memory.reset(g.session_id)

        return jsonify({
            "success":    True,
            "session_id": g.session_id,
            "message":    f"Memory cleared for session {g.session_id}",
        }), 200

    except Exception as exc:
        log.exception("reset() error")
        return _error("Internal server error.", 500, str(exc))


# ── GET /stats ────────────────────────────────────────────────────────────────

@app.route("/stats", methods=["GET"])
@require_ready
@log_request
def stats():
    """
    Aggregated monitoring stats from the SQLite log.

    Query params:
        ?hours=24    (default 24 — look-back window)

    Response 200:
        {
            "success": true,
            "window_hours": 24,
            "stats": {
                "total_queries":  42,
                "avg_time":       3.1,
                "avg_confidence": 0.81,
                "success_rate":   97.6
            }
        }
    """
    try:
        hours = int(request.args.get("hours", 24))
        if hours < 1 or hours > 720:
            return _error("'hours' must be between 1 and 720.", 400)

        data = _orch.monitor.get_stats(hours=hours)
        if data:
            return jsonify({
                "success":      True,
                "window_hours": hours,
                "stats":        data,
            }), 200
        return jsonify({
            "success": False,
            "message": "No stats available yet for this window.",
        }), 404

    except Exception as exc:
        log.exception("stats() error")
        return _error("Internal server error.", 500, str(exc))


# ── GET /history ──────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
@require_ready
@with_session
@log_request
def history():
    """
    Return conversation history for the current session.

    Response 200:
        {
            "success":    true,
            "session_id": "abc-123",
            "history": [
                { "query": "How many PTO days?",
                  "answer": "Employees get 15 days..." },
                ...
            ],
            "context_summary": "Q: How many PTO days?"
        }
    """
    try:
        mem = _get_or_create_memory(g.session_id)
        # Sanitise any stored answers in history too
        clean_history = [
            {
                "query":  entry.get("query", ""),
                "answer": sanitise_answer(entry.get("answer", "")),
            }
            for entry in mem.history
        ]
        return jsonify({
            "success":         True,
            "session_id":      g.session_id,
            "history":         clean_history,
            "context_summary": mem.context_summary,
        }), 200

    except Exception as exc:
        log.exception("history() error")
        return _error("Internal server error.", 500, str(exc))


# ── 404 / 405 handlers ────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return _error("Method not allowed.", 405)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # threaded=True is REQUIRED for concurrent SSE connections
    app.run(host="0.0.0.0", port=8080, threaded=True, debug=False)
