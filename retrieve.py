"""
Phase 3: hybrid retrieval (dense + sparse, combined with RRF).

Loads the indexes built in Phase 2, and lets you type questions in the
terminal to see which chunks get retrieved for each one.

Run with: python retrieve.py
"""

import pickle
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "ask_my_docs"
BM25_FILE = "bm25_index.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_PER_METHOD = 10  # how many results to pull from each method before fusing
RRF_K = 60              # standard constant used in the RRF formula
FINAL_TOP_N = 5         # how many fused results to show/return


def load_everything():
    client = QdrantClient(path=QDRANT_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    with open(BM25_FILE, "rb") as f:
        bm25_data = pickle.load(f)

    return client, model, bm25_data["bm25"], bm25_data["chunks"]


def dense_search(client, model, query, top_k):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    # Return list of chunk_ids in ranked order.
    return [r.payload["chunk_id"] for r in results]


def sparse_search(bm25, chunks, query, top_k):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # Pair each chunk with its score, sort by score descending.
    ranked = sorted(
        zip(chunks, scores), key=lambda pair: pair[1], reverse=True
    )
    return [chunk["chunk_id"] for chunk, score in ranked[:top_k]]


def reciprocal_rank_fusion(dense_ids, sparse_ids, k=RRF_K):
    """Combine two ranked lists into one fused ranking using RRF."""
    scores = {}

    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    for rank, chunk_id in enumerate(sparse_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    # Sort chunk_ids by their fused score, highest first.
    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, score in fused]


def hybrid_retrieve(query, client, model, bm25, chunks, chunk_lookup):
    dense_ids = dense_search(client, model, query, TOP_K_PER_METHOD)
    sparse_ids = sparse_search(bm25, chunks, query, TOP_K_PER_METHOD)
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)

    top_chunks = [chunk_lookup[cid] for cid in fused_ids[:FINAL_TOP_N]]
    return top_chunks


if __name__ == "__main__":
    print("Loading indexes and model...")
    client, model, bm25, chunks = load_everything()
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    print("Ready. Type a question (or 'quit' to exit).\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        results = hybrid_retrieve(query, client, model, bm25, chunks, chunk_lookup)

        print(f"\nTop {len(results)} retrieved chunks:\n")
        for i, chunk in enumerate(results, 1):
            preview = chunk["text"][:200].replace("\n", " ")
            print(f"[{i}] from '{chunk['source_doc']}' (chunk {chunk['chunk_index_in_doc']})")
            print(f"    {preview}...\n")
