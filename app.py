import json
import streamlit as st
import chromadb
import torch
import requests
from sentence_transformers import SentenceTransformer
from config import (
    EMBED_MODEL,
    COLLECTION_NAME,
    CHROMA_PATH,
    DOMAINS,
    OLLAMA_BASE_URL,
    LLM_MODEL,
    RAG_TOP_K,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medical Knowledge Assistant", page_icon="🩺", layout="wide")

# ── System Prompt (STRICT + STRUCTURED) ──────────────────────────────────────
SYSTEM_PROMPT = """You are a strict medical retrieval assistant.

RULES:
- Answer ONLY using provided context
- DO NOT use prior knowledge
- DO NOT assume missing info
- If exact answer is not found → say:
"I cannot answer this based on the provided medical literature."

STRICT DIAGNOSIS RULES:
- Choose ONLY ONE most supported diagnosis
- Do NOT list multiple possibilities
- Do NOT infer beyond context

FORMAT:
Diagnosis:
Reasoning:
Evidence: (Source X: file.pdf)

SAFETY:
- Do NOT give treatment advice
- Always add disclaimer at end:
*Disclaimer: I am an AI, not a doctor. Educational use only.*
"""

# ── Domain Keywords ───────────────────────────────────────────────────────────
CARDIAC_KEYWORDS = {"heart", "cardiac", "ecg", "arrhythmia", "tachycardia"}
GYNAE_KEYWORDS = {"pregnancy", "pregnant", "preeclampsia", "uterine"}

def detect_domain(q):
    q = q.lower()
    if any(k in q for k in CARDIAC_KEYWORDS):
        return "cardiac"
    if any(k in q for k in GYNAE_KEYWORDS):
        return "gynae"
    return None

# ── Cached Resources ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device)

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

# ── Retrieval ────────────────────────────────────────────────────────────────
def retrieve_chunks(query, model, collection, domain=None, k=4):
    instruction = "Represent this question for searching relevant passages: "
    embedding = model.encode(instruction + query, normalize_embeddings=True)

    where = {"domain": domain} if domain else None

    return collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas"]
    )

def build_context(results):
    parts = []
    sources = []

    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        parts.append(f"[Source {i}: {meta.get('source')}]\n{doc}")

        sources.append({
            "id": i,
            "file": meta.get("source"),
            "domain": meta.get("domain")
        })

    return "\n\n---\n\n".join(parts), sources

# ── Ollama Generation (FIXED) ────────────────────────────────────────────────
def generate_answer_streaming(query, context):
    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024
        }
    }

    with requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token

# ── UI ───────────────────────────────────────────────────────────────────────
def main():
    st.title("🩺 Medical Knowledge Assistant")
    st.markdown("RAG-powered Clinical Assistant (Local LLM)")
    st.divider()

    model = load_model()
    collection = get_collection()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        domain_filter = st.selectbox("Domain", ["Auto"] + DOMAINS)
        k = st.slider("Top-K", 1, 10, RAG_TOP_K)

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Enter medical query...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        domain = detect_domain(user_input) if domain_filter == "Auto" else domain_filter

        with st.chat_message("assistant"):
            with st.spinner("Retrieving knowledge..."):
                results = retrieve_chunks(user_input, model, collection, domain, k)

                if not results["documents"] or not results["documents"][0]:
                    response = "I cannot answer this based on the provided medical literature."
                    st.markdown(response)
                    return

                context, sources = build_context(results)

            # STREAM RESPONSE
            placeholder = st.empty()
            full_response = ""

            try:
                for token in generate_answer_streaming(user_input, context):
                    full_response += token
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                with st.expander("Sources"):
                    for s in sources:
                        st.write(f"Source {s['id']} - {s['file']} ({s['domain']})")

            except Exception as e:
                full_response = f"❌ Error: {str(e)}"
                placeholder.error(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

if __name__ == "__main__":
    main()