import os
import json
import hashlib
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from config import (
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    CHROMA_PATH,
    CHUNKS_FILE,
    EMBED_BATCH_SIZE,
)
from logger import setup_logger

logger = setup_logger(__name__)


def compute_text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def get_existing_document_hashes(collection):
    try:
        result = collection.get(include=["metadatas"])
        if result and result.get("metadatas"):
            return set(
                m.get("text_hash", "")
                for m in result["metadatas"]
                if m.get("text_hash")
            )
    except Exception as e:
        logger.warning(f"Could not fetch existing hashes: {e}")
    return set()


def deduplicate_chunks(chunks, existing_hashes=None):
    if existing_hashes is None:
        existing_hashes = set()

    unique_chunks = []
    seen_hashes = set()

    for chunk in chunks:
        text_hash = compute_text_hash(chunk["text"])

        if text_hash in existing_hashes or text_hash in seen_hashes:
            logger.debug(f"Skipping duplicate: {chunk['source'][:30]}...")
            continue

        seen_hashes.add(text_hash)
        chunk["text_hash"] = text_hash
        unique_chunks.append(chunk)

    return unique_chunks


def load_chunks():
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        logger.info("Run 'python pdf_to_chunks.py' first to generate chunks.")
        return None

    try:
        with open(CHUNKS_FILE, "r") as f:
            chunks = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chunks file: {e}")
        return None

    if not chunks:
        logger.warning("No chunks found in file")
        return None

    return chunks


def main():
    logger.info("Starting embedding storage process")

    chunks = load_chunks()
    if chunks is None:
        return

    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    logger.info("Model loaded")

    logger.info("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    existing_hashes = get_existing_document_hashes(collection)
    logger.info(f"Found {len(existing_hashes)} existing documents in collection")

    new_chunks = deduplicate_chunks(chunks, existing_hashes)
    logger.info(f"New unique chunks to embed: {len(new_chunks)}")

    if not new_chunks:
        logger.info("No new chunks to add. Database is up to date.")
        return

    instruction = "Represent this medical passage for retrieval: "
    texts = [instruction + chunk["text"] for chunk in new_chunks]

    logger.info("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    ids = [str(i) for i in range(len(new_chunks))]

    batch_size = 1000
    total = len(new_chunks)

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        collection.upsert(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end].tolist(),
            documents=[chunk["text"] for chunk in new_chunks[i:batch_end]],
            metadatas=[
                {
                    "source": chunk["source"],
                    "domain": chunk["domain"],
                    "text_hash": chunk.get(
                        "text_hash", compute_text_hash(chunk["text"])
                    ),
                }
                for chunk in new_chunks[i:batch_end]
            ],
        )
        logger.info(f"Stored batch {i} → {batch_end}")

    logger.info(f"Successfully stored {total} new embeddings")


if __name__ == "__main__":
    main()
