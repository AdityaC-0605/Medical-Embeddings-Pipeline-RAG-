# embed_and_store.py

import json
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from config import *

CHUNKS_FILE = "data/chunks.json"


def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print("Model loaded ✓")

    instruction = "Represent this medical passage for retrieval: "
    texts = [instruction + chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Initializing persistent ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [str(i) for i in range(len(chunks))]

    print("Storing embeddings in batches...")

    batch_size = 1000
    total = len(chunks)

    for i in range(0, total, batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=[chunk["text"] for chunk in chunks[i:i + batch_size]],
            metadatas=[{
                "source": chunk["source"],
                "domain": chunk["domain"]
            } for chunk in chunks[i:i + batch_size]]
        )

        print(f"Stored batch {i} → {min(i + batch_size, total)}")

    print("\n✅ All embeddings stored successfully in ChromaDB")


if __name__ == "__main__":
    main()