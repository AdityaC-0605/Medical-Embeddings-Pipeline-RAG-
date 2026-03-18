import json                          # FIX: moved to top-level imports
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

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🏥", layout="wide")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise medical assistant powered by a retrieval-augmented generation system.

Your role:
- Answer medical questions using ONLY the provided context from the retrieved medical literature.
- Use clear, professional medical language.

RESPONSE FORMAT (STRICT):
- Be CONCISE and DIRECT. Do not use conversational filler.
- Synthesize points from multiple sources instead of listing them one by one.
- ALWAYS cite sources inline using this exact format: (Source 1: filename.pdf).
- Do NOT generate a "References" or "Sources used" section at the end.

Important:
- Never provide personal medical advice or diagnosis.
- Put the citation inline immediately after the claim."""

# ── Domain keywords (kept in sync with chatbot.py) ────────────────────────────

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
    """Drop minority-domain chunks when one domain has ≥60% of results."""
    docs = results["documents"][0]
    if len(docs) <= 1:
        return results

    counts = {}
    for m in results["metadatas"][0]:
        d = m.get("domain", "unknown")
        counts[d] = counts.get(d, 0) + 1

    dominant        = max(counts, key=counts.get)
    dominance_ratio = counts[dominant] / len(docs)

    if dominance_ratio >= 0.6 and len(counts) > 1:
        keep = [i for i, m in enumerate(results["metadatas"][0])
                if m.get("domain") == dominant]
        return {
            "documents": [[docs[i] for i in keep]],
            "metadatas": [[results["metadatas"][0][i] for i in keep]],
        }

    return results


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device), device


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(query, model, collection, domain=None, k=5):
    instruction = "Represent this question for searching relevant passages: "
    embedding   = model.encode(instruction + query, normalize_embeddings=True)
    where       = {"domain": domain} if domain else None

    return collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas"],
    )


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(results):
    parts   = []
    sources = []

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), start=1
    ):
        text = doc[:800] + ("..." if len(doc) > 800 else "")
        parts.append(
            f"[Source {i}: {meta.get('source')} | Domain: {meta.get('domain')}]\n{text}"
        )
        entry = f"{meta.get('domain')} - {meta.get('source')}"
        if entry not in sources:
            sources.append(entry)

    return "\n\n---\n\n".join(parts), sources


# ── Streaming LLM via /api/chat ───────────────────────────────────────────────

def generate_answer_streaming(query, context):
    user_prompt = (
        f"Medical Context (retrieved from indexed literature):\n\n{context}\n\n"
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
            {"role": "user",   "content": user_prompt},
        ],
        "stream": True,
        "options": {"temperature": 0.3, "num_predict": 1024},
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)          # uses top-level import now
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.title("🏥 Medical RAG Chatbot")
    st.markdown("Ask medical questions grounded in your documents")

    try:
        model, device = load_model()
        collection    = get_collection()
        count         = collection.count()
    except Exception as e:
        st.error(f"Startup error: {e}")
        return

    with st.sidebar:
        st.success(f"Connected: {count} chunks")
        st.info(f"Device: {device}")
        st.info(f"LLM: {LLM_MODEL}")

        st.header("Settings")
        domain_filter = st.selectbox("Domain", ["Auto-detect"] + DOMAINS)
        k = st.slider("Top-K", 1, 10, RAG_TOP_K)

    query = st.text_input("Ask your medical question:")

    if not query:
        return

    # Determine domain
    if domain_filter == "Auto-detect":
        active_domain = detect_domain(query)
        domain_label  = f"Auto → {active_domain}" if active_domain else "Auto → all"
    else:
        active_domain = domain_filter
        domain_label  = domain_filter

    st.caption(f"🏷️ Domain: {domain_label}")

    with st.spinner("Retrieving relevant chunks..."):
        results = retrieve_chunks(query, model, collection, active_domain, k)

        if not results["documents"] or not results["documents"][0]:
            st.warning("No results found. Try rephrasing or changing the domain.")
            return

        if not active_domain:
            results = filter_majority_domain(results)

        context, sources = build_context(results)

    # Stream the answer
    st.subheader("🧠 Answer")
    answer_placeholder = st.empty()
    full_answer = ""

    try:
        for token in generate_answer_streaming(query, context):
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")

        answer_placeholder.markdown(full_answer)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Ollama. Run: `ollama serve`")
        return
    except Exception as e:
        st.error(f"❌ Generation error: {e}")
        return

    st.subheader("📚 Sources")
    for s in sources:
        st.write(f"- {s}")

    with st.expander("🔍 Retrieved Context"):
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), start=1
        ):
            st.markdown(f"**--- Chunk {i} ---**")
            st.markdown(f"*{meta.get('domain')} — {meta.get('source')}*")
            st.write(doc)


if __name__ == "__main__":
    main()