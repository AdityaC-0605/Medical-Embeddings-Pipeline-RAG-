import chromadb
import torch
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, COLLECTION_NAME, CHROMA_PATH
from logger import setup_logger

logger = setup_logger(__name__)


def query_database(query_text, model, collection, n_results=5):
    query_instruction = "Represent this medical question for retrieval: "

    query_embedding = model.encode(
        query_instruction + query_text, normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=n_results
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
        user_question = input("Question: ")

        if user_question.lower() == "quit":
            logger.info("User exited query interface")
            break

        if not user_question.strip():
            continue

        results = query_database(user_question, model, collection)

        if not results["documents"] or not results["documents"][0]:
            print("No results found.")
            continue

        print("\nRetrieved Chunks:\n")

        for i in range(len(results["documents"][0])):
            print(f"--- Result {i + 1} ---")
            print("Source:", results["metadatas"][0][i].get("source", "Unknown"))
            print("Domain:", results["metadatas"][0][i].get("domain", "Unknown"))
            print(results["documents"][0][i][:500])
            print("\n")


if __name__ == "__main__":
    main()
