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

# ── Page Configuration & CSS ──────────────────────────────────────────────────
st.set_page_config(page_title="Medical Knowledge Assistant", page_icon="🩺", layout="wide")

# Custom CSS to remove Streamlit branding and polish the UI
st.markdown("""
    <style>
    /* Hide Streamlit default header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Slightly tone down the sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Style the source tags to look like badges */
    .source-badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: #e9ecef;
        color: #495057;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise medical assistant powered by a retrieval-augmented generation system.

Your role:
- Answer medical questions using ONLY the provided context from the retrieved medical literature.
- Use clear, professional medical language.

RESPONSE FORMAT (STRICT):
- Be CONCISE and DIRECT. 
- NEVER start your response with conversational filler like "Based on the retrieved medical literature...". Start answering immediately.
- Synthesize points from multiple sources instead of listing them one by one.
- ALWAYS cite sources inline using this exact format: (Source 1: filename.pdf).
- Do NOT generate a "References" or "Sources used" section at the end.

REFUSAL PROTOCOL (CRITICAL):
- If the provided context does NOT contain the exact answer, you MUST refuse to answer. 
- NO DEMOGRAPHIC ASSUMPTIONS: If the text is about a specific demographic (e.g., pregnant women, children) and the user's prompt is generic (e.g., "A patient"), you MUST refuse to answer. Do not assume the user's patient matches the textbook's demographic.
- STRICT SYMPTOM MATCHING: If the user lists specific symptoms and the retrieved text describes a condition that does NOT include those specific symptoms, you MUST refuse to diagnose. Do not force a diagnosis based on a partial match.
- State exactly: "I cannot answer this based on the provided medical literature."
- NEVER use your general knowledge to fill in gaps. NEVER hallucinate a citation.

SAFETY GUARDRAILS (CRITICAL):
- Never provide a definitive personal medical diagnosis or suggest treatments for an active emergency.
- You MUST use softening language (e.g., "The literature suggests this presentation aligns with...", "These symptoms are commonly associated with...").
- You MUST append this exact disclaimer to the very end of your response, formatted in italics: "*Disclaimer: I am an AI, not a doctor. This information is for educational purposes based on retrieved literature and is not a clinical diagnosis.*"
"""

# ── Domain keywords ───────────────────────────────────────────────────────────
CARDIAC_KEYWORDS = {
    "heart", "cardiac", "cardiovascular", "myocardial", "infarction", "angina",
    "coronary", "arrhythmia", "atrial", "ventricular", "fibrillation",
    "valve", "valvular", "aortic", "mitral", "pericarditis", "endocarditis",
    "cardiomyopathy", "heart failure", "hf", "acs", "atherosclerosis",
    "chest pain", "dyspnea", "orthopnea", "edema", "palpitation", "syncope",
    "leg swelling", "ankle swelling", "jugular", "ecg", "electrocardiogram",
    "stemi", "nstemi", "troponin", "bnp", "ejection fraction", "tachycardia",
    "bradycardia", "hypertension", "hypotension", "murmur", "pulmonary edema",
    "shortness of breath", "breathlessness", "fatigue", "swelling",
}

GYNAE_KEYWORDS = {
    "pregnancy", "pregnant", "obstetric", "gynae", "gynecology", "pcos",
    "polycystic", "endometriosis", "ectopic", "preeclampsia", "eclampsia",
    "ovarian", "uterine", "cervical", "menstrual", "period", "contraception",
    "fertility", "infertility", "miscarriage", "placenta", "fetal", "trimester",
    "labor", "delivery", "cesarean", "postpartum", "prenatal", "antenatal",
    "ovulation", "fallopian", "fibroids", "hysterectomy", "vaginal",
    "amenorrhea", "menopause", "gestational", "maternal", "hirsutism",
}

def detect_domain(question: str) -> str | None:
    q = question.lower()
    cardiac_score = sum(1 for kw in CARDIAC_KEYWORDS if kw in q)
    gynae_score   = sum(1 for kw in GYNAE_KEYWORDS   if kw in q)

    if cardiac_score > 0 and gynae_score == 0:
        return "cardiac"
    elif gynae_score > 0 and cardiac_score == 0:
        return "gynae"
    elif cardiac_score > 0 and gynae_score > 0:
        if cardiac_score > gynae_score:
            return "cardiac"
        elif gynae_score > cardiac_score:
            return "gynae"
    return None

def filter_majority_domain(results: dict) -> dict:
    docs = results["documents"][0]
    if len(docs) <= 1:
        return results

    counts = {}
    for m in results["metadatas"][0]:
        d = m.get("domain", "unknown")
        counts[d] = counts.get(d, 0) + 1

    dominant = max(counts, key=counts.get)
    dominance_ratio = counts[dominant] / len(docs)

    if dominance_ratio >= 0.6 and len(counts) > 1:
        keep = [i for i, m in enumerate(results["metadatas"][0])
                if m.get("domain") == dominant]
        return {
            "documents": [[docs[i] for i in keep]],
            "metadatas": [[results["metadatas"][0][i] for i in keep]],
        }

    return results

# ── Cached Resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device), device

@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)

# ── Core Logic ────────────────────────────────────────────────────────────────
def retrieve_chunks(query, model, collection, domain=None, k=5):
    instruction = "Represent this question for searching relevant passages: "
    embedding = model.encode(instruction + query, normalize_embeddings=True)
    where = {"domain": domain} if domain else None

    return collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas"],
    )

def build_context(results):
    parts = []
    sources = []

    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        text = doc[:800] + ("..." if len(doc) > 800 else "")
        parts.append(f"[Source {i}: {meta.get('source')} | Domain: {meta.get('domain')}]\n{text}")
        
        entry = {"id": i, "domain": meta.get('domain'), "file": meta.get('source')}
        if entry not in sources:
            sources.append(entry)

    return "\n\n---\n\n".join(parts), sources

def generate_answer_streaming(query, context):
    user_prompt = (
        f"Medical Context:\n{context}\n\n"
        f"───────────────────────────────────\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"1. Answer directly and concisely using ONLY the context above.\n"
        f"2. Cite sources inline like: (Source N: filename.pdf)\n"
        f"3. Do NOT output a references list at the bottom."
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }

    with requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break

# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    # Application Header
    st.title("🩺 Medical Knowledge Assistant")
    st.markdown("Retrieval-Augmented Diagnostic & Clinical Reference Tool")
    st.divider()

    # System Initialization
    try:
        model, device = load_model()
        collection = get_collection()
    except Exception as e:
        st.error("System Initialization Failed. Please check the backend services.")
        st.code(str(e))
        return

    # Session State Management
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar Interface
    with st.sidebar:
        st.header("System Status")
        col1, col2 = st.columns(2)
        col1.metric("Indexed Docs", collection.count())
        col2.metric("Compute", device.upper())
        
        st.markdown(f"**LLM Engine:** `{LLM_MODEL}`")
        st.divider()
        
        st.header("Query Configuration")
        domain_filter = st.selectbox("Routing Logic", ["Auto-detect"] + DOMAINS, index=0)
        
        with st.expander("Advanced Settings"):
            k = st.slider("Retrieval Depth (Top-K)", 1, 10, RAG_TOP_K)
            
        st.divider()
        if st.button("Clear Session History", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Empty State / Onboarding
    if len(st.session_state.messages) == 0:
        st.info("👋 **Welcome to the Medical RAG Dashboard.**\n\n"
                "Please enter a clinical query below. The system will route your question "
                "to the appropriate domain (Cardiac/Gynae), retrieve relevant literature, "
                "and synthesize a cited response.")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Retrieved Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"<span class='source-badge'>Source {s['id']}</span> **{s['file']}** ({s['domain']})", unsafe_allow_html=True)

    # Input Capture
    user_input = st.chat_input("Enter clinical symptoms or medical query...")

    if user_input:
        # Append and display user input
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Domain Resolution
        active_domain = detect_domain(user_input) if domain_filter == "Auto-detect" else domain_filter

        # Pipeline Execution
        with st.chat_message("assistant"):
            with st.spinner(f"Querying knowledge base... [{active_domain or 'cross-domain'}]"):
                results = retrieve_chunks(user_input, model, collection, active_domain, k)

                if not results["documents"] or not results["documents"][0]:
                    response = "No relevant medical literature found for this query in the current index."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})
                    return

                if not active_domain:
                    results = filter_majority_domain(results)

                context, sources = build_context(results)

            # Context augmentation with short-term memory
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
            enhanced_context = f"Previous conversation:\n{history_text}\n---\nCurrent retrieved context:\n{context}"

            # Stream generation
            response_placeholder = st.empty()
            full_response = ""

            try:
                for token in generate_answer_streaming(user_input, enhanced_context):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                # Render Sources directly under the response
                with st.expander("View Retrieved Sources"):
                    for s in sources:
                         st.markdown(f"<span class='source-badge'>Source {s['id']}</span> **{s['file']}** ({s['domain']})", unsafe_allow_html=True)

            except Exception as e:
                full_response = f"Generation Error: {str(e)}"
                response_placeholder.error(full_response)

        # Save state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response, "sources": sources}
        )

if __name__ == "__main__":
    main()