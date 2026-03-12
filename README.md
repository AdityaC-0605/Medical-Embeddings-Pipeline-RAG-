# Medical Embeddings Pipeline

A lightweight Python pipeline for processing medical documents. It extracts, cleans, and chunks text from medical PDFs, then generates dense vector embeddings. These embeddings are stored in a local vector database ([ChromaDB](https://docs.trychroma.com)) for efficient retrieval-augmented generation (RAG) queries.

## 🚀 Features

- **Robust PDF Extraction:** Uses `PyMuPDF` to read text, intelligently ignoring references and citations.
- **Smart Text Chunking:** Splits text sequentially by sentences, enforcing limits with a default chunk size of 900 characters and a 120-character overlap.
- **Sentence Transformers:** Employs the `BAAI/bge-large-en-v1.5` model, optimized with instruction prefixes for retrieval and querying.
- **Apple Silicon (MPS) Hardware Acceleration:** Auto-detects and uses `torch.backends.mps` for faster processing on M-series Macs, if available.
- **Persistent Local DB:** Stores collections efficiently offline in a dedicated `chroma_db` folder.
- **Interactive Query Engine:** Offers a simple CLI to retrieve the top semantically similar documents.
- **Web UI:** A Streamlit-based interface for easier querying and domain filtering.
- **Incremental Processing:** Processes only new PDFs, skipping those already handled.
- **Duplicate Detection:** Prevents duplicate embeddings using SHA256 hashing.
- **Evaluation Metrics:** Includes a built-in script for measuring retrieval performance.

## 📂 Project Structure

```
medical_embeddings_project/
├── config.py             # Global constants for models, dimensions, and chunk sizes
├── logger.py             # Logging configuration
├── pdf_to_chunks.py      # Cleans and splits raw PDFs into chunks.json
├── embed_and_store.py    # Generates embeddings and stores in Chroma DB
├── query.py              # Interactive CLI script for querying the database
├── chatbot.py            # RAG chatbot (retrieval + LLM answer generation)
├── app.py                # Streamlit web UI for querying
├── evaluate.py           # Evaluation metrics script
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore patterns
├── data/                 # Raw data directory
│   ├── cardiac/          # Cardiac-related PDFs
│   ├── gynae/            # Gynaecology-related PDFs
│   └── chunks.json       # Generated text chunks
└── chroma_db/            # Local persistent vector DB
```

## 🛠️ Installation & Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd medical_embeddings_project
   ```

2. **Create and activate a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ How to Run

### Step 1: Add your documents
Place your medical `.pdf` files into the `data/cardiac/` or `data/gynae/` directories.

### Step 2: Extract and Chunk Text
This step extracts text, cleans noise (like URLs and DOIs), and chunks the sentences:
```bash
# Incremental mode (skips already processed files)
python pdf_to_chunks.py

# Full reprocessing (processes all PDFs)
python pdf_to_chunks.py --full
```

### Step 3: Embed and Store Chunks into ChromaDB
Generate the embedding vectors and ingest them into ChromaDB:
```bash
python embed_and_store.py
```
- Automatically detects duplicates using SHA256 hashing.
- Only adds new, unique chunks to the database.

### Step 4: Query the Embeddings

**CLI Interface (retrieval only):**
```bash
python query.py
```

**RAG Chatbot (retrieval + LLM-generated answers):**

Requires [Ollama](https://ollama.com) running locally:
```bash
# Start Ollama (if not already running)
ollama serve

# Pull a model (first time only)
ollama pull llama3.1:8b

# Run the chatbot
python chatbot.py

# With domain filter
python chatbot.py --domain cardiac

# With a different model
python chatbot.py --model mistral:7b

# Adjust number of retrieved chunks
python chatbot.py --top-k 8
```

**Web UI (Streamlit):**
```bash
streamlit run app.py
```

### Step 5: Evaluate Retrieval Performance
```bash
python evaluate.py
```

## ⚙️ Configuration

All settings can be customized in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBED_MODEL` | BAAI/bge-large-en-v1.5 | Sentence transformer model used for embeddings. |
| `CHUNK_SIZE` | 900 | Maximum chunk size in characters. |
| `CHUNK_OVERLAP` | 120 | Overlap between consecutive chunks in characters. |
| `EMBED_BATCH_SIZE` | 32 | Batch size for generating embeddings. |
| `DOMAINS` | [cardiac, gynae] | Supported medical domains. |

### Environment Variables

You can also configure paths using environment variables:
```bash
export DATA_DIR=data
export CHROMA_PATH=chroma_db
export CHUNKS_FILE=data/chunks.json
export LOG_LEVEL=INFO
export LOG_FILE=app.log

# LLM / Chatbot settings
export LLM_MODEL=llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434
export LLM_TEMPERATURE=0.3
export LLM_MAX_TOKENS=1024
export RAG_TOP_K=5
```

## 📊 Evaluation

The evaluation script uses predefined queries with ground truth sources to compute:
- **Recall@K**: The fraction of relevant documents retrieved within the top K results.
- **Precision@K**: The accuracy of the documents retrieved within the top K results.
- **MRR**: Mean Reciprocal Rank, measuring the position of the first relevant document.
- **NDCG**: Normalized Discounted Cumulative Gain, assessing the ranking quality.