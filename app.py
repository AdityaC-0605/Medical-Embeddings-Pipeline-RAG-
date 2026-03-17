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


# -----------------------------
# Load model & DB
# -----------------------------
@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device), device


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


# -----------------------------
# Retrieval
# -----------------------------
def retrieve_chunks(query, model, collection, domain=None, k=5):
    instruction = "Represent this medical question for retrieval: "
    embedding = model.encode(instruction + query, normalize_embeddings=True)

    where = {"domain": domain} if domain else None

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas"],
    )

    return results


# -----------------------------
# Build context
# -----------------------------
def build_context(results):
    context = ""
    sources = set()

    for i in range(len(results["documents"][0])):
        chunk = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        context += f"\n\n{chunk}"
        sources.add(f"{metadata.get('domain')} - {metadata.get('source')}")

    return context[:4000], list(sources)


# -----------------------------
# LLM (Ollama)
# -----------------------------
def generate_answer(query, context):
    prompt = f"""
You are a medical assistant. Answer ONLY from the provided context.

Context:
{context}

Question:
{query}

Answer clearly and medically.
"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )

    return response.json()["response"]


# -----------------------------
# UI
# -----------------------------
def main():
    st.title("🏥 Medical RAG Chatbot")
    st.markdown("Ask medical questions grounded in your documents")

    # Load
    try:
        model, device = load_model()
        collection = get_collection()
        count = collection.count()

        st.sidebar.success(f"Connected: {count} chunks")
        st.sidebar.info(f"Device: {device}")
        st.sidebar.info(f"LLM: {LLM_MODEL}")

    except Exception as e:
        st.error(f"Error: {e}")
        return

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        domain_filter = st.selectbox("Domain", ["All"] + DOMAINS)
        k = st.slider("Top-K", 1, 10, RAG_TOP_K)

    # Input
    query = st.text_input("Ask your medical question:")

    if query:
        with st.spinner("Thinking..."):

            domain = None if domain_filter == "All" else domain_filter

            # Step 1: Retrieval
            results = retrieve_chunks(query, model, collection, domain, k)

            if not results["documents"] or not results["documents"][0]:
                st.warning("No results found")
                return

            # Step 2: Context
            context, sources = build_context(results)

            # Step 3: LLM Answer
            answer = generate_answer(query, context)

        # Output
        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for s in sources:
            st.write(f"- {s}")

        with st.expander("🔍 Retrieved Context"):
            for i in range(len(results["documents"][0])):
                st.write(f"--- Chunk {i+1} ---")
                st.write(results["documents"][0][i])


if __name__ == "__main__":
    main()