import os
from pathlib import Path

# ---------------------------------------
# 🔧 Disable Chroma telemetry (fix crashes)
# ---------------------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ---------------------------------------
# 📦 Embedding Config
# ---------------------------------------
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
EMBED_DIM = 1024

# ---------------------------------------
# 🧠 Vector DB Config
# ---------------------------------------
COLLECTION_NAME = "medical_bge"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

DOMAINS = ["cardiac", "gynae"]

EMBED_BATCH_SIZE = 32
QUERY_BATCH_SIZE = 1000

# ---------------------------------------
# 📁 Paths
# ---------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
CHUNKS_FILE = os.getenv("CHUNKS_FILE", "data/chunks.json")

# ---------------------------------------
# 📝 Logging
# ---------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# ---------------------------------------
# 🤖 LLM CONFIG (IMPORTANT)
# ---------------------------------------

# 🔥 For normal app (UI / query)
LLM_MODEL = "mistral:7b"

# 🔥 For RAGAS evaluation (fast + stable)
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "mistral:7b")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Retrieval
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# ---------------------------------------
# 📁 Utils
# ---------------------------------------
def get_project_root():
    return Path(__file__).parent


def ensure_dirs():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)


# ✅ Ensure dirs exist before anything runs
ensure_dirs()