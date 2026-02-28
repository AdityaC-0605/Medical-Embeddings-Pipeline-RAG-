# pdf_to_chunks.py

import os
import re
import json
import fitz
from tqdm import tqdm
from config import CHUNK_SIZE, CHUNK_OVERLAP

REFERENCES_PATTERN = re.compile(r'^\s*References\s*$', re.MULTILINE)

fitz.TOOLS.mupdf_display_errors(False)

DATA_DIR = "data"
OUTPUT_FILE = "data/chunks.json"


def clean_text(text):
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove DOIs
    text = re.sub(r'doi:\S+', '', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove citation numbers [12]
    text = re.sub(r'\[\d+\]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path, filetype="pdf") as doc:
            text = ""

            for page in doc:
                page_text = page.get_text("text")

                # Stop before references section (match standalone heading only)
                if REFERENCES_PATTERN.search(page_text):
                    # Keep text before the "References" heading on this page
                    match = REFERENCES_PATTERN.search(page_text)
                    text += page_text[:match.start()]
                    break

                text += page_text

            return text

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def chunk_text(text, chunk_size, overlap):
    # Split into sentences first for cleaner boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence exceeds chunk_size, save current and start new
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunk = current_chunk.strip()
            if len(chunk) > 300 and not chunk.isnumeric():
                chunks.append(chunk)

            # Keep overlap by finding sentences that fit within overlap size
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            # Start new chunk from the last sentence boundary within overlap
            last_break = overlap_text.rfind('. ')
            if last_break != -1:
                current_chunk = overlap_text[last_break + 2:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    # Don't forget the last chunk
    chunk = current_chunk.strip()
    if len(chunk) > 300 and not chunk.isnumeric():
        chunks.append(chunk)

    return chunks


def main():
    all_chunks = []

    for domain in ["cardiac", "gynae"]:
        folder_path = os.path.join(DATA_DIR, domain)
        if not os.path.exists(folder_path):
            continue

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        print(f"\n📂 {domain.upper()} — {len(pdf_files)} PDFs")

        for pdf_file in tqdm(pdf_files):
            pdf_path = os.path.join(folder_path, pdf_file)

            text = extract_text_from_pdf(pdf_path)
            if not text:
                continue

            text = clean_text(text)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": pdf_file,
                    "domain": domain
                })

    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print("\n✅ Done.")
    print(f"Saved {len(all_chunks)} clean chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()