import os
from pathlib import Path

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024

COLLECTION_NAME = "medical_bge"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

DOMAINS = ["cardiac", "gynae"]

EMBED_BATCH_SIZE = 32
QUERY_BATCH_SIZE = 1000

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "data/chunks.json")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")


def get_project_root():
    return Path(__file__).parent


def ensure_dirs():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
