import chromadb
import torch
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, COLLECTION_NAME, CHROMA_PATH
from logger import setup_logger

logger = setup_logger(__name__)


def query_database(query_text, model, collection, domain_filter=None, n_results=5):
    query_instruction = "Represent this question for searching relevant passages: "

    query_embedding = model.encode(
        query_instruction + query_text, normalize_embeddings=True
    )

    where_clause = {"domain": domain_filter} if domain_filter else None

    # FIX: explicitly specify include= so metadatas and distances are always
    # returned regardless of ChromaDB version defaults.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    return results


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    logger.info("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Collection '{COLLECTION_NAME}' not found: {e}")
        print(f"\n❌ Collection '{COLLECTION_NAME}' not found.")
        print("Run 'python embed_and_store.py' first to generate embeddings.")
        return

    logger.info(f"Connected to collection: {COLLECTION_NAME}")
    count = collection.count()
    logger.info(f"Total documents in collection: {count}")

    print("\nMedical RAG Query Interface")
    print(f"Database contains {count} documents")
    print("Type 'quit' to exit.\n")

    while True:
        user_question = input("\nQuestion: ")

        if user_question.lower() == "quit":
            logger.info("User exited query interface")
            break

        if not user_question.strip():
            continue

        domain_filter_input = input("Domain filter (Press Enter for all): ")
        domain_filter = domain_filter_input.strip().lower() if domain_filter_input.strip() else None

        results = query_database(user_question, model, collection, domain_filter)

        if not results["documents"] or not results["documents"][0]:
            print("No results found.")
            continue

        print("\nRetrieved Chunks:\n")

        for i in range(len(results["documents"][0])):
            similarity = round(1 - results["distances"][0][i], 4)
            print(f"--- Result {i + 1} ---")
            print("Source  :", results["metadatas"][0][i].get("source", "Unknown"))
            print("Domain  :", results["metadatas"][0][i].get("domain", "Unknown"))
            print("Sim     :", similarity)
            print(results["documents"][0][i][:500])
            print()


if __name__ == "__main__":
    main()