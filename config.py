import os
from pathlib import Path

# Disable ChromaDB telemetry globally to prevent ClientStartEvent/capture() bugs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024

COLLECTION_NAME = "medical_bge"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

DOMAINS = ["cardiac", "gynae"]

EMBED_BATCH_SIZE = 32
QUERY_BATCH_SIZE = 1000

DATA_DIR    = os.getenv("DATA_DIR",    "data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "data/chunks.json")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = os.getenv("LOG_FILE",  "app.log")

# LLM Configuration (Ollama)
LLM_MODEL        = os.getenv("LLM_MODEL",        "llama3.1:8b")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS",    "1024"))
RAG_TOP_K        = int(os.getenv("RAG_TOP_K",         "5"))


def get_project_root():
    return Path(__file__).parent


def ensure_dirs():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)


# FIX: ensure_dirs() is now called so data/ and chroma_db/ always exist
# before any other module tries to read or write to them.
ensure_dirs()