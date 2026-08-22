import streamlit as st
import requests
import time
import uuid

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_BASE   = "http://98.82.42.19:5000"
QUERY_URL  = f"{API_BASE}/query"
HEALTH_URL = f"{API_BASE}/health"
RESET_URL  = f"{API_BASE}/reset"

TIMEOUT = 120

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{uuid.uuid4()}"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Agentic RAG")
    st.caption("Policy & Healthcare Assistant")

    st.markdown("---")

    # 🔍 Health check
    st.subheader("System Status")

    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.ok:
            data = r.json()
            if data.get("ready"):
                st.success("✅ Model Ready")
                st.caption(f"Uptime: {data.get('uptime_seconds', 0)} sec")
            else:
                st.warning("⏳ Loading model...")
        else:
            st.error("⚠️ API issue")
    except:
        st.error("❌ Cannot connect to API")

    st.markdown("---")

    # 🔁 Reset chat
    if st.button("🧹 Reset Conversation"):
        try:
            requests.post(
                RESET_URL,
                headers={"X-Session-Id": st.session_state.session_id},
                timeout=5
            )
        except:
            pass

        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # 💡 Suggestions
    st.subheader("💡 Try asking")

    suggestions = [
        "How many PTO days do employees get?",
        "What is the remote work policy?",
        "What health insurance is available?",
        "Explain Policy C-901"
    ]

    for s in suggestions:
        if st.button(s):
            st.session_state.input_box = s

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.title("🧠 Agentic RAG Assistant")
st.caption("Ask questions about policies, healthcare docs, compliance rules")

# ─────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# INPUT BOX
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ─────────────────────────────────────────
    # CALL API
    # ─────────────────────────────────────────
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

        try:
            start = time.time()

            resp = requests.post(
                QUERY_URL,
                json={
                    "question": user_input,
                    "enable_memory": True,
                    "enable_self_correction": True
                },
                headers={
                    "X-Session-Id": st.session_state.session_id
                },
                timeout=TIMEOUT
            )

            elapsed = round(time.time() - start, 2)

            # ───────────── ERROR HANDLING ─────────────
            if resp.status_code == 503:
                answer = "⏳ Model is loading. Try again in a few seconds."

            elif resp.status_code != 200:
                answer = f"⚠️ Error {resp.status_code}: {resp.text}"

            else:
                data = resp.json()

                answer      = data.get("answer", "No answer.")
                citations   = data.get("citations", "")
                confidence  = data.get("confidence", None)

                # ───────── CLEAN ANSWER ─────────
                answer_clean = answer.split("\nSources:")[0].strip()

                final = answer_clean

                # ───────── ADD SOURCES ─────────
                if citations:
                    final += "\n\n📚 **Sources**\n"
                    for line in citations.split("\n"):
                        if line.strip():
                            final += f"- {line.strip()}\n"

                # ───────── CONFIDENCE ─────────
                if confidence is not None:
                    pct = int(confidence * 100)
                    final += f"\n\n🎯 Confidence: **{pct}%**"

                    if pct < 60:
                        final += "\n⚠️ Low confidence — verify manually."

                # ───────── META ─────────
                final += f"\n\n⏱ Response time: {elapsed}s"

                answer = final

        except Exception as e:
            answer = f"❌ Connection error: {str(e)}"

        placeholder.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })