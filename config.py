import os
from pathlib import Path

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ---------------- CORE SETTINGS ----------------
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "medical_bge"
CHROMA_PATH = "chroma_db"

# Available domains for the UI selectbox
DOMAINS = ["cardiac", "gynae"]

# LLM Settings
LLM_MODEL = "llama3.1:8b"        # Used by the Chatbot
RAGAS_LLM_MODEL = "llama3.1:8b"  # Used by the Evaluation script
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval Settings
RAG_TOP_K = 5

# ---------------- DIRECTORIES ----------------
def ensure_dirs():
    Path("data").mkdir(exist_ok=True)
    Path(CHROMA_PATH).mkdir(exist_ok=True)

ensure_dirs()