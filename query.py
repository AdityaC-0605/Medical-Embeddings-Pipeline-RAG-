# query.py

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from config import *

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"\n❌ Collection '{COLLECTION_NAME}' not found.")
        print("Run 'python embed_and_store.py' first to generate embeddings.")
        return

    print("\nMedical RAG Query Interface")
    print("Type 'quit' to exit.\n")

    while True:
        user_question = input("Question: ")

        if user_question.lower() == "quit":
            break

        query_instruction = "Represent this medical question for retrieval: "

        query_embedding = model.encode(
            query_instruction + user_question,
            normalize_embeddings=True
        )

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )

        print("\nRetrieved Chunks:\n")

        for i in range(len(results["documents"][0])):
            print(f"--- Result {i+1} ---")
            print("Source:", results["metadatas"][0][i]["source"])
            print("Domain:", results["metadatas"][0][i]["domain"])
            print(results["documents"][0][i][:500])
            print("\n")


if __name__ == "__main__":
    main()