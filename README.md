# Medical Embeddings Pipeline

A lightweight Python-based retrieval pipeline designed for medical documents. This project extracts, cleans, chunks, and creates dense vector embeddings for medical PDFs, subsequently storing them in a local vector database ([ChromaDB](https://docs.trychroma.com/)) for quick retrieval-augmented generation (RAG) queries.

## 🚀 Features

- **Robust PDF Extraction:** Uses `PyMuPDF` to read text, specifically ignoring references and citations.
- **Smart Text Chunking:** Splits text by sentences sequentially, cleanly enforcing limits (default 900 chunk size with 120 character overlap).
- **Sentence Transformers:** Employs the `BAAI/bge-large-en-v1.5` embeddings model, optimized with instruction prefixes for retrieval/querying.
- **Apple Silicon (MPS) Hardware Acceleration:** Auto-detects and utilizes `torch.backends.mps` for faster processing on M-series Macs if available.
- **Persistent Local DB:** Stores collections efficiently offline inside a persistent `chroma_db` folder.
- **Interactive Query Engine:** Provides a simple CLI to fetch the top contextually relevant documents based on semantic similarity.

## 📂 Project Structure

```
medical_embeddings_project/
├── config.py             # Global constants for models, dimensions, and chunk sizes
├── pdf_to_chunks.py      # Cleans and splits raw PDFs located in the data directory into a chunks.json file
├── embed_and_store.py    # Generates dense embeddings for chunks and pushes batch arrays to Chroma DB
├── query.py              # Interactive CLI script for querying the database
├── requirements.txt      # Project dependencies
├── data/                 # Raw data directory
│   ├── cardiac/          # Place Cardiac-related PDFs here
│   ├── gynae/            # Place Gynaecology-related PDFs here
│   └── chunks.json       # Generated text chunks intermediate file
└── chroma_db/            # Generated Local persistent vector DB store
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
Place your medical `.pdf` files into `data/cardiac/` or `data/gynae/` domains (or respectively configure domains inside `pdf_to_chunks.py`).

### Step 2: Extract and Chunk Text
Extract text, clean noise (URLs, DOIs, excessive whitespaces) and chunk the sentences:
```bash
python pdf_to_chunks.py
```
*This script automatically outputs `data/chunks.json`.*

### Step 3: Embed and Store chunks into ChromaDB
Generate the embedding vectors and ingest them into a ChromaDB database running entirely locally:
```bash
python embed_and_store.py
```

### Step 4: Query the Embeddings
Fire up the CLI user interface to question your vector database.
```bash
python query.py
```
- Type your question and the tool retrieves the top 5 semantically matching chunks.
- Type `quit` to exit.

## 💡 Customization

- To adjust the Embedding model, update the `EMBED_MODEL` value inside `config.py`.
- To tune Retrieval granularity, alter `CHUNK_SIZE` and `CHUNK_OVERLAP` inside `config.py`.
