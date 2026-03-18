import os
import re
import json
import fitz
from tqdm import tqdm
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR, CHUNKS_FILE, DOMAINS
from logger import setup_logger

logger = setup_logger(__name__)

REFERENCES_PATTERN = re.compile(r"^\s*References\s*$", re.MULTILINE)

fitz.TOOLS.mupdf_display_errors(False)


def clean_text(text):
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"doi:\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[.]{3,}", " ", text)
    text = re.sub(r"[-]{3,}", " ", text)
    text = re.sub(r"[_]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_quality_chunk(text, min_length=100):
    """Check if a chunk has meaningful content (not just dots, dashes, numbers, etc.)"""
    if len(text) < min_length:
        return False
    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) == 0 or alpha_chars / len(text) < 0.3:
        return False
    return True


def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path, filetype="pdf") as doc:
            text = ""
            for page in doc:
                page_text = page.get_text("text")
                if REFERENCES_PATTERN.search(page_text):
                    match = REFERENCES_PATTERN.search(page_text)
                    text += page_text[: match.start()]
                    break
                text += page_text
            return text
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return ""


def chunk_text(text, chunk_size, overlap):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunk = current_chunk.strip()
            if is_quality_chunk(chunk, min_length=100):
                chunks.append(chunk)

            overlap_text = (
                current_chunk[-overlap:]
                if len(current_chunk) > overlap
                else current_chunk
            )
            last_break = overlap_text.rfind(". ")
            if last_break != -1:
                current_chunk = overlap_text[last_break + 2 :] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    # FIX: was min_length=300, which silently dropped valid short concluding
    # paragraphs. Now consistent with the body chunk threshold of 100.
    chunk = current_chunk.strip()
    if is_quality_chunk(chunk, min_length=100):
        chunks.append(chunk)

    return chunks


def get_existing_chunks():
    if os.path.exists(CHUNKS_FILE):
        try:
            with open(CHUNKS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Invalid chunks.json, starting fresh")
            return []
    return []


def get_processed_files(existing_chunks):
    return set(chunk.get("source") for chunk in existing_chunks if "source" in chunk)


def process_pdfs(incremental=True):
    existing_chunks = get_existing_chunks() if incremental else []
    processed_files = get_processed_files(existing_chunks) if incremental else set()

    all_chunks = existing_chunks.copy()
    new_files_count = 0

    for domain in DOMAINS:
        folder_path = os.path.join(DATA_DIR, domain)
        if not os.path.exists(folder_path):
            logger.warning(f"Domain folder not found: {folder_path}")
            continue

        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        logger.info(f"{domain.upper()} — {len(pdf_files)} PDFs found")

        for pdf_file in tqdm(pdf_files, desc=f"Processing {domain}"):
            if incremental and pdf_file in processed_files:
                logger.debug(f"Skipping already processed: {pdf_file}")
                continue

            pdf_path = os.path.join(folder_path, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_file}")
                continue

            text = clean_text(text)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            for chunk in chunks:
                all_chunks.append({"text": chunk, "source": pdf_file, "domain": domain})

            new_files_count += 1
            logger.info(f"Processed: {pdf_file} ({len(chunks)} chunks)")

    return all_chunks, new_files_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to chunks")
    parser.add_argument(
        "--full", action="store_true", help="Process all PDFs (ignore incremental mode)"
    )
    args = parser.parse_args()

    incremental = not args.full

    logger.info(f"Starting PDF processing (incremental={incremental})")

    all_chunks, new_files_count = process_pdfs(incremental=incremental)

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    logger.info(f"Done. Processed {new_files_count} new files.")
    logger.info(f"Saved {len(all_chunks)} total chunks to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()