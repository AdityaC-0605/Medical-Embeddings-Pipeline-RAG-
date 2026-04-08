import os
from pathlib import Path

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Embedding
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# Vector DB
COLLECTION_NAME = "medical_bge"
CHROMA_PATH = "chroma_db"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:8b"

# App
DOMAINS = ["cardiac", "gynae"]
RAG_TOP_K = 4

# Directories
def ensure_dirs():
    Path("data").mkdir(exist_ok=True)
    Path(CHROMA_PATH).mkdir(exist_ok=True)

ensure_dirs()