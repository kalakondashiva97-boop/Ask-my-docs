"""
Phase 2: build the dense (vector) and sparse (BM25) indexes.

Reads chunks.json (from Phase 1) and produces:
  1. A Qdrant collection (dense/embedding search), stored locally in
     ./qdrant_data — no separate server needed.
  2. A BM25 index (keyword search), saved to bm25_index.pkl.

Run with: python build_index.py
(First run will take a minute or two — it downloads an embedding model.)
"""

import json
import pickle

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

CHUNKS_FILE = "chunks.json"
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "ask_my_docs"
BM25_FILE = "bm25_index.pkl"

# A small, fast, well-regarded open-source embedding model.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # this model always outputs vectors of this size


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dense_index(chunks, model):
    print("Embedding chunks (this may take a minute)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    client = QdrantClient(path=QDRANT_PATH)

    # Recreate the collection fresh each time we run this script.
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload=chunks[i],  # store all chunk metadata alongside the vector
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Dense index built: {len(points)} vectors stored in {QDRANT_PATH}")


def build_sparse_index(chunks):
    # BM25 needs tokenized (word-split) text, lowercased for consistency.
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # We save both the BM25 model and the chunks list (BM25 only knows
    # about word statistics, not the original text/metadata, so we need
    # to keep the chunk list alongside it to map results back later).
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Sparse index built: saved to {BM25_FILE}")


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    build_dense_index(chunks, model)
    build_sparse_index(chunks)

    print("\nPhase 2 indexing complete.")
