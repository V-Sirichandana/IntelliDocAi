import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import faiss
import numpy as np
from datetime import datetime
import json
import os
import hashlib
import re
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="IntelliDoc AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.sidebar-section {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #888;
    margin: 1.2rem 0 0.4rem 0;
}
.file-card {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.file-name { font-weight: 600; color: #c9d1d9; word-break: break-all; }
.file-meta { color: #888; font-size: 0.74rem; margin-top: 2px; }
.hist-item {
    background: #161622;
    border-left: 3px solid #4f46e5;
    border-radius: 0 6px 6px 0;
    padding: 6px 10px;
    margin-bottom: 5px;
    font-size: 0.78rem;
    color: #aaa;
}
.hist-q {
    color: #c9d1d9; font-weight: 500;
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; max-width: 200px;
}
.hist-ts { color: #555; font-size: 0.68rem; margin-top: 2px; }
.session-item {
    background: #12121c;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 0.78rem;
}
.session-title { color: #c9d1d9; font-weight: 600; font-size: 0.8rem; }
.session-meta  { color: #555; font-size: 0.68rem; margin-top: 2px; }
.chat-ts { font-size: 0.68rem; color: #555; margin-top: 2px; }
.conf-high { color: #4ade80; font-size: 0.7rem; font-weight: 600; }
.conf-mid  { color: #facc15; font-size: 0.7rem; font-weight: 600; }
.conf-low  { color: #f87171; font-size: 0.7rem; font-weight: 600; }
.token-badge {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.68rem;
    color: #888;
    margin-top: 4px;
}
.welcome-card {
    background: linear-gradient(135deg, #1e1e2e, #16213e);
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- GROQ CLIENT ----------------

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- SESSIONS DIRECTORY ----------------

SESSIONS_DIR = "intellidoc_sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# ---------------- SESSION STATE ----------------

for key, default in [
    ("chat_history",    []),
    ("vector_store",    None),
    ("processed_files", []),
    ("file_meta",       {}),
    ("session_id",      None),
    ("session_name",    ""),
    ("total_tokens",    0),
    ("active_view",     "chat"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- MODELS ----------------

@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    embed  = SentenceTransformer("all-MiniLM-L6-v2")
    rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embed, rerank

embedding_model, reranker = load_models()

# ---------------- PERSISTENCE HELPERS ----------------

def make_session_id(file_names: list) -> str:
    key = "_".join(sorted(file_names))
    return hashlib.md5(key.encode()).hexdigest()[:10]

def session_path(sid: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{sid}.json")

def save_session(sid, name, file_names, chat_history):
    data = {
        "id":           sid,
        "name":         name,
        "file_names":   file_names,
        "chat_history": chat_history,
        "saved_at":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "turn_count":   len(chat_history),
    }
    with open(session_path(sid), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def list_sessions():
    sessions = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(SESSIONS_DIR, fname), "r") as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
    return sessions

def delete_session(sid):
    p = session_path(sid)
    if os.path.exists(p):
        os.remove(p)

def export_chat_txt(session_name, file_names, chat_history):
    lines = [
        "=" * 60,
        "  IntelliDoc AI — Session Export",
        f"  Session : {session_name}",
        f"  Docs    : {', '.join(file_names)}",
        f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60, "",
    ]
    for i, turn in enumerate(chat_history, 1):
        lines += [
            f"[Q{i}] {turn.get('timestamp','')}",
            f"You: {turn['question']}",
            "",
            f"AI : {turn['answer']}",
            "-" * 60, "",
        ]
    return "\n".join(lines)

# ---------------- CORE RAG FUNCTIONS ----------------

def build_vector_store(uploaded_files):
    """Extract text, chunk per file, build FAISS + BM25 indexes."""
    chunks         = []
    chunk_metadata = []
    file_names     = []
    file_meta      = {}

    for file in uploaded_files:
        file_names.append(file.name)
        pdf_reader = PdfReader(file)
        pages      = pdf_reader.pages
        full_text  = ""

        for page in pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"

        file_meta[file.name] = {
            "pages":  len(pages),
            "words":  len(full_text.split()),
            "upload": datetime.now().strftime("%H:%M"),
        }

        splitter    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        file_chunks = splitter.split_text(full_text)

        for chunk in file_chunks:
            chunks.append(chunk)
            chunk_metadata.append({"source": file.name})

    # FAISS dense index
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # BM25 sparse index
    tokenized = [re.sub(r"[^a-z0-9 ]", " ", c.lower()).split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    return {
        "index":        index,
        "bm25":         bm25,
        "chunks":       chunks,
        "metadata":     chunk_metadata,
        "file_names":   file_names,
        "total_chunks": len(chunks),
    }, file_meta


def tokenize_bm25(text: str):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranked_list in rankings:
        for rank, idx in enumerate(ranked_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def retrieve_context(query, vector_store, top_k=15, final_k=5):
    """
    Hybrid Search pipeline:
      1. Dense  — FAISS cosine similarity
      2. Sparse — BM25 keyword
      3. RRF    — fuse both ranked lists
      4. CrossEncoder — re-rank fused candidates
    Returns: list of dicts {text, source}, confidence string
    """
    index    = vector_store["index"]
    bm25     = vector_store["bm25"]
    chunks   = vector_store["chunks"]
    metadata = vector_store["metadata"]
    n        = len(chunks)

    # 1. Dense search
    q_emb = embedding_model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
    _, I_dense    = index.search(q_emb, k=min(top_k, n))
    dense_ranking = [int(i) for i in I_dense[0] if i < n]

    # 2. Sparse BM25 search
    q_tokens       = tokenize_bm25(query)
    bm25_scores    = bm25.get_scores(q_tokens)
    sparse_ranking = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    # 3. RRF fusion
    fused        = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    fused_sorted = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]

    # Build candidate list — plain dicts, NOT tuples
    candidates = [
        {"text": chunks[i], "source": metadata[i]["source"]}
        for i in fused_sorted
    ]

    # 4. CrossEncoder re-ranking
    pairs        = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    # Sort candidates by score (descending), keeping them as dicts
    ranked_candidates = [
        c for _, c in sorted(
            zip(rerank_scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
    ]

    top_chunks   = ranked_candidates[:final_k]           # list of dicts
    rerank_score = float(max(rerank_scores)) if len(rerank_scores) else 0.0

    if rerank_score > 3:
        confidence = "high"
    elif rerank_score > 0:
        confidence = "mid"
    else:
        confidence = "low"

    return top_chunks, confidence   # top_chunks = [{"text":..., "source":...}, ...]


def build_groq_messages(chat_history, context, query):
    system_prompt = """You are IntelliDoc AI, an intelligent PDF document assistant.

Answer questions ONLY from the provided document context.

Rules:
- Be accurate and direct.
- ALWAYS mention which PDF/document the answer came from.
- If information comes from multiple PDFs, clearly mention each source.
- Cite source file names in the answer.
- When comparing multiple documents, clearly distinguish between them.
- Use conversation history to understand follow-up questions.
- If the answer is not in the context, say: "Answer not found in the uploaded documents."
- Never hallucinate or add outside information.
- Use bullet points or numbered lists when listing or comparing.
- Keep answers concise but complete."""

    messages = [{"role": "system", "content": system_prompt}]

    for turn in chat_history[-6:]:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({"role": "user", "content": f"""Document context:
---
{context}
---
Question: {query}"""})

    return messages


def get_followup_suggestions(answer, context):
    try:
        resp = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": (
                    "Based on this AI answer and document context, "
                    "suggest exactly 3 short follow-up questions a user might ask next.\n"
                    "Return ONLY a JSON array of 3 strings. No explanation, no markdown.\n\n"
                    f"Answer: {answer[:500]}\n"
                    f"Context snippet: {context[:300]}"
                )
            }],
            model="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions[:3]]
    except Exception:
        pass
    return []

# ================================================================
#  SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("## 📚 IntelliDoc AI")
    st.divider()

    # View toggle
    col_a, col_b = st.columns(2)
    if col_a.button("💬 Chat", use_container_width=True,
                    type="primary" if st.session_state.active_view == "chat" else "secondary"):
        st.session_state.active_view = "chat"
        st.rerun()
    if col_b.button("🕘 Sessions", use_container_width=True,
                    type="primary" if st.session_state.active_view == "history" else "secondary"):
        st.session_state.active_view = "history"
        st.rerun()

    st.divider()

    # Upload
    st.markdown('<div class="sidebar-section">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Loaded documents
    if st.session_state.vector_store:
        vs   = st.session_state.vector_store
        meta = st.session_state.file_meta

        st.markdown('<div class="sidebar-section">Loaded Documents</div>', unsafe_allow_html=True)
        for name in vs["file_names"]:
            m = meta.get(name, {})
            st.markdown(f"""
<div class="file-card">
  <div class="file-name">📄 {name}</div>
  <div class="file-meta">
    {m.get('pages','?')} pages &nbsp;·&nbsp;
    {m.get('words','?'):,} words &nbsp;·&nbsp;
    ⏱ {m.get('upload','')}
  </div>
</div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("Docs",   len(vs["file_names"]))
        c2.metric("Chunks", vs["total_chunks"])

        if st.session_state.total_tokens > 0:
            st.markdown(
                f'<div class="token-badge">🔢 ~{st.session_state.total_tokens:,} tokens used</div>',
                unsafe_allow_html=True
            )

    # Current session history list
    if st.session_state.chat_history and st.session_state.active_view == "chat":
        st.markdown('<div class="sidebar-section">This Session</div>', unsafe_allow_html=True)

        for i, turn in enumerate(reversed(st.session_state.chat_history)):
            idx     = len(st.session_state.chat_history) - i
            q_short = turn["question"][:52] + ("…" if len(turn["question"]) > 52 else "")
            st.markdown(f"""
<div class="hist-item">
  <div class="hist-q">Q{idx}: {q_short}</div>
  <div class="hist-ts">{turn.get('timestamp','')}</div>
</div>""", unsafe_allow_html=True)

        sname = st.text_input(
            "Save session as…",
            value=st.session_state.session_name or
                  f"Session {datetime.now().strftime('%b %d %H:%M')}",
            key="sname_input",
        )

        sc1, sc2 = st.columns(2)
        if sc1.button("💾 Save", use_container_width=True):
            sid = st.session_state.session_id or make_session_id(
                st.session_state.vector_store["file_names"]
            )
            st.session_state.session_id   = sid
            st.session_state.session_name = sname
            save_session(sid, sname,
                         st.session_state.vector_store["file_names"],
                         st.session_state.chat_history)
            st.success("Saved!")

        if sc2.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.total_tokens = 0
            st.rerun()

        if st.session_state.vector_store:
            export_txt = export_chat_txt(
                st.session_state.session_name or "session",
                st.session_state.vector_store["file_names"],
                st.session_state.chat_history
            )
            st.download_button(
                "⬇️ Export Chat (.txt)",
                data=export_txt,
                file_name=f"intellidoc_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Suggested questions (before first message)
    if (st.session_state.vector_store
            and not st.session_state.chat_history
            and st.session_state.active_view == "chat"):
        st.markdown('<div class="sidebar-section">Try asking…</div>', unsafe_allow_html=True)
        for s in [
            "Summarize all uploaded documents",
            "What are the key topics covered?",
            "Compare the main points across documents",
            "What conclusions are drawn?",
            "List important dates or figures mentioned",
        ]:
            if st.button(s, use_container_width=True, key=f"sug_{s}"):
                st.session_state["_prefill"] = s
                st.rerun()

# ================================================================
#  PROCESS UPLOADS
# ================================================================

if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])
    if current_names != st.session_state.processed_files:
        with st.spinner("Processing PDFs and building vector index…"):
            vs, meta = build_vector_store(uploaded_files)
            st.session_state.vector_store    = vs
            st.session_state.file_meta       = meta
            st.session_state.processed_files = current_names
            st.session_state.chat_history    = []
            st.session_state.total_tokens    = 0
            st.session_state.session_id      = make_session_id(current_names)
            st.session_state.session_name    = ""
        st.success(f"✅ {len(current_names)} document(s) indexed — ready to chat!")

# ================================================================
#  VIEW: SAVED SESSIONS
# ================================================================

if st.session_state.active_view == "history":
    st.markdown("## 🕘 Saved Sessions")
    sessions = list_sessions()

    if not sessions:
        st.info("No saved sessions yet. Chat with documents and click **💾 Save** in the sidebar.")
    else:
        for sess in sessions:
            with st.expander(
                f"📂 **{sess['name']}** — {sess['turn_count']} Q&As · {sess['saved_at']}",
                expanded=False
            ):
                st.caption(f"Documents: {', '.join(sess['file_names'])}")

                for i, turn in enumerate(sess["chat_history"], 1):
                    with st.chat_message("user"):
                        st.write(f"**Q{i}:** {turn['question']}")
                        st.markdown(
                            f'<div class="chat-ts">{turn.get("timestamp","")}</div>',
                            unsafe_allow_html=True
                        )
                    with st.chat_message("assistant"):
                        st.write(turn["answer"])

                d1, d2 = st.columns(2)
                d1.download_button(
                    "⬇️ Export",
                    data=export_chat_txt(sess["name"], sess["file_names"], sess["chat_history"]),
                    file_name=f"intellidoc_{sess['id']}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"exp_{sess['id']}"
                )
                if d2.button("🗑️ Delete", use_container_width=True, key=f"del_{sess['id']}"):
                    delete_session(sess["id"])
                    st.rerun()

# ================================================================
#  VIEW: CHAT
# ================================================================

elif st.session_state.active_view == "chat":

    if not st.session_state.vector_store:
        st.markdown("""
<div class="welcome-card">
  <h3>👋 Welcome to IntelliDoc AI</h3>
  <p style="color:#888; margin:0.5rem 0 1.5rem">
    Upload one or more PDF files in the sidebar to get started.
  </p>
  <p style="color:#666; font-size:0.85rem">
    Ask questions · Compare documents · Multi-turn chat · Save &amp; revisit sessions
  </p>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown("## 💬 Chat with your Documents")

        conf_labels = {
            "high": '<span class="conf-high">● High confidence</span>',
            "mid":  '<span class="conf-mid">● Medium confidence</span>',
            "low":  '<span class="conf-low">● Low confidence</span>',
        }

        # Render existing turns
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
                st.markdown(
                    f'<div class="chat-ts">{turn.get("timestamp","")}</div>',
                    unsafe_allow_html=True
                )
            with st.chat_message("assistant"):
                st.markdown(conf_labels.get(turn.get("confidence","mid"), ""),
                            unsafe_allow_html=True)
                st.write(turn["answer"])
                st.markdown(
                    f'<div class="chat-ts">{turn.get("timestamp","")}</div>',
                    unsafe_allow_html=True
                )
                if turn.get("followups"):
                    st.markdown("**Suggested follow-ups:**")
                    fu_cols = st.columns(len(turn["followups"]))
                    for j, sug in enumerate(turn["followups"]):
                        if fu_cols[j].button(sug, key=f"fu_{turn.get('timestamp','')}_{j}"):
                            st.session_state["_prefill"] = sug
                            st.rerun()

        # Input
        prefill = st.session_state.pop("_prefill", "")
        query   = st.chat_input("Ask a question about your documents…")
        if not query and prefill:
            query = prefill

        if query:
            ts = datetime.now().strftime("%H:%M")

            with st.chat_message("user"):
                st.write(query)
                st.markdown(f'<div class="chat-ts">{ts}</div>', unsafe_allow_html=True)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):

                    # Retrieve — returns list of dicts {text, source}
                    top_chunks, confidence = retrieve_context(
                        query, st.session_state.vector_store
                    )

                    # Build context string with source labels
                    context = "\n\n".join(
                        f"[Source: {c['source']}]\n{c['text']}"
                        for c in top_chunks          # c is a dict, not a tuple
                    )

                    messages = build_groq_messages(
                        st.session_state.chat_history, context, query
                    )

                    completion = client.chat.completions.create(
                        messages=messages,
                        model="llama-3.1-8b-instant",
                        temperature=0.2,
                        max_tokens=1024,
                    )
                    answer   = completion.choices[0].message.content
                    usage    = completion.usage
                    tok_used = usage.total_tokens if usage else 0

                    followups = get_followup_suggestions(answer, context)

                st.session_state.total_tokens += tok_used

                st.markdown(conf_labels.get(confidence, ""), unsafe_allow_html=True)
                st.write(answer)
                st.markdown(
                    f'<div class="chat-ts">{ts} · ~{tok_used} tokens</div>',
                    unsafe_allow_html=True
                )

                if followups:
                    st.markdown("**Suggested follow-ups:**")
                    fu_cols = st.columns(len(followups))
                    for j, sug in enumerate(followups):
                        if fu_cols[j].button(sug, key=f"new_fu_{j}_{ts}"):
                            st.session_state["_prefill"] = sug
                            st.rerun()

            # Save turn to session state
            st.session_state.chat_history.append({
                "question":   query,
                "answer":     answer,
                "timestamp":  ts,
                "confidence": confidence,
                "followups":  followups,
                "tokens":     tok_used,
            })

            # Auto-save to disk
            sid = st.session_state.session_id or make_session_id(
                st.session_state.vector_store["file_names"]
            )
            st.session_state.session_id = sid
            save_session(
                sid,
                st.session_state.session_name or f"Auto {datetime.now().strftime('%b %d %H:%M')}",
                st.session_state.vector_store["file_names"],
                st.session_state.chat_history,
            )

            st.rerun()