import streamlit as st
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, COLLECTION_NAME, CHROMA_PATH, DOMAINS

st.set_page_config(page_title="Medical RAG Query", page_icon="🏥", layout="wide")


@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device), device


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def query_database(query_text, model, collection, domain_filter=None, n_results=5):
    query_instruction = "Represent this medical question for retrieval: "

    query_embedding = model.encode(
        query_instruction + query_text, normalize_embeddings=True
    )

    where = {"domain": domain_filter} if domain_filter else None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=where if where else None,
        include=["metadatas", "documents"],
    )

    return results


def main():
    st.title("🏥 Medical RAG Query Interface")
    st.markdown("Query medical documents using semantic search")

    try:
        model, device = load_model()
        collection = get_collection()

        doc_count = collection.count()
        st.sidebar.success(f"Connected - {doc_count} documents indexed")
        st.sidebar.info(f"Device: {device}")
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info("Run 'python embed_and_store.py' first to generate embeddings.")
        return

    with st.sidebar:
        st.header("Filters")
        domain_filter = st.selectbox("Filter by Domain", options=["All"] + DOMAINS)

        n_results = st.slider("Number of results", 1, 10, 5)

        st.header("Stats")
        for domain in DOMAINS:
            try:
                count = collection.count(where={"domain": domain})
                st.metric(domain.capitalize(), count)
            except Exception:
                st.metric(domain.capitalize(), 0)

    query = st.text_input(
        "Ask a medical question:",
        placeholder="e.g., What are the symptoms of heart failure?",
    )

    if query:
        with st.spinner("Searching..."):
            domain = None if domain_filter == "All" else domain_filter
            results = query_database(query, model, collection, domain, n_results)

        if (
            not results.get("documents")
            or not results["documents"][0]
            or not results["documents"][0][0]
        ):
            st.warning("No results found.")
        else:
            st.success(f"Found {len(results['documents'][0])} results")

            for i in range(len(results["documents"][0])):
                metadata = (
                    results["metadatas"][0][i]
                    if results["metadatas"] and results["metadatas"][0]
                    else {}
                )
                doc_text = results["documents"][0][i] if results["documents"][0] else ""

                with st.expander(
                    f"Result {i + 1}: {metadata.get('source', 'Unknown')}",
                    expanded=True,
                ):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown("**Domain:**")
                        st.markdown("**Source:**")
                    with col2:
                        st.markdown(metadata.get("domain", "Unknown"))
                        st.markdown(metadata.get("source", "Unknown"))

                    st.markdown("---")
                    st.markdown(doc_text)


if __name__ == "__main__":
    main()
