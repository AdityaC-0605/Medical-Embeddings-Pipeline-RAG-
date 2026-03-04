# Medical Embeddings Pipeline

A lightweight Python-based retrieval pipeline designed for medical documents. This project extracts, cleans, chunks, and creates dense vector embeddings for medical PDFs, subsequently storing them in a local vector database ([ChromaDB](https://docs.trychroma.com/)) for quick retrieval-augmented generation (RAG) queries.

## 🚀 Features

- **Robust PDF Extraction:** Uses `PyMuPDF` to read text, specifically ignoring references and citations.
- **Smart Text Chunking:** Splits text by sentences sequentially, cleanly enforcing limits (default 900 chunk size with 120 character overlap).
- **Sentence Transformers:** Employs the `BAAI/bge-large-en-v1.5` embeddings model, optimized with instruction prefixes for retrieval/querying.
- **Apple Silicon (MPS) Hardware Acceleration:** Auto-detects and utilizes `torch.backends.mps` for faster processing on M-series Macs if available.
- **Persistent Local DB:** Stores collections efficiently offline inside a persistent `chroma_db` folder.
- **Interactive Query Engine:** Provides a simple CLI to fetch the top contextually relevant documents based on semantic similarity.
- **Web UI:** Streamlit-based interface for easier querying with domain filtering.
- **Incremental Processing:** Only processes new PDFs, skipping already processed files.
- **Duplicate Detection:** Prevents duplicate embeddings using SHA256 hashing.
- **Evaluation Metrics:** Built-in evaluation script for measuring retrieval performance.

## 📂 Project Structure

```
medical_embeddings_project/
├── config.py             # Global constants for models, dimensions, and chunk sizes
├── logger.py             # Logging configuration
├── pdf_to_chunks.py      # Cleans and splits raw PDFs into chunks.json
├── embed_and_store.py    # Generates embeddings and stores in Chroma DB
├── query.py              # Interactive CLI script for querying the database
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

1. **Clone/Navigate to the directory**:
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
Place your medical `.pdf` files into `data/cardiac/` or `data/gynae/` directories.

### Step 2: Extract and Chunk Text
Extract text, clean noise (URLs, DOIs, excessive whitespaces) and chunk the sentences:
```bash
# Incremental mode (skip already processed files)
python pdf_to_chunks.py

# Full reprocessing (process all PDFs)
python pdf_to_chunks.py --full
```

### Step 3: Embed and Store chunks into ChromaDB
Generate the embedding vectors and ingest them into ChromaDB:
```bash
python embed_and_store.py
```
- Automatically detects duplicates using SHA256 hashing
- Only adds new unique chunks to the database

### Step 4: Query the Embeddings

**CLI Interface:**
```bash
python query.py
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
|-----------|---------|-------------|
| `EMBED_MODEL` | BAAI/bge-large-en-v1.5 | Sentence transformer model |
| `CHUNK_SIZE` | 900 | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | 120 | Overlap between chunks |
| `EMBED_BATCH_SIZE` | 32 | Batch size for embedding generation |
| `DOMAINS` | [cardiac, gynae] | Supported medical domains |

### Environment Variables

You can also configure paths via environment variables:
```bash
export DATA_DIR=data
export CHROMA_PATH=chroma_db
export CHUNKS_FILE=data/chunks.json
export LOG_LEVEL=INFO
export LOG_FILE=app.log
```

## 📊 Evaluation

The evaluation script uses predefined queries with ground truth sources to compute:
- **Recall@K**: Fraction of relevant documents retrieved
- **Precision@K**: Accuracy of retrieved documents
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
