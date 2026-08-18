"""
Phase 1, Step B: split documents into chunks.

Reads every .txt file in data/, splits each into overlapping word-based
chunks, and saves the result as a single chunks.json file. Every chunk
carries metadata (which document it came from, and its position) so we
can trace answers back to sources later.

Run with: python chunk_documents.py
"""

import os
import json

DATA_DIR = "data"
OUTPUT_FILE = "chunks.json"

CHUNK_SIZE_WORDS = 300   # roughly 300 words per chunk
OVERLAP_WORDS = 50       # last 50 words of one chunk repeat at the start of the next


def chunk_text(text: str, chunk_size: int, overlap: int):
    """Split text into a list of overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break
        start = end - overlap  # step forward, but re-include the overlap

    return chunks


def build_all_chunks():
    all_chunks = []
    chunk_counter = 0

    filenames = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt"))

    for filename in filenames:
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        doc_name = filename.replace(".txt", "").replace("_", " ")
        pieces = chunk_text(text, CHUNK_SIZE_WORDS, OVERLAP_WORDS)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "source_doc": doc_name,
                "source_file": filename,
                "chunk_index_in_doc": i,
                "text": piece,
            })
            chunk_counter += 1

        print(f"{filename}: {len(pieces)} chunks")

    return all_chunks


if __name__ == "__main__":
    chunks = build_all_chunks()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Saved to {OUTPUT_FILE}")
