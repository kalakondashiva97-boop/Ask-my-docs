"""
Phase 4: cross-encoder reranking.

Takes the fused hybrid retrieval results from Phase 3 and re-scores them
with a cross-encoder model for more precise relevance ranking.

Run with: python rerank.py
"""

import pickle
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "ask_my_docs"
BM25_FILE = "bm25_index.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_PER_METHOD = 10   # candidates pulled from each retrieval method
RRF_K = 60
CANDIDATES_FOR_RERANK = 10  # how many fused candidates the reranker sees
FINAL_TOP_N = 5             # how many reranked results to keep


def load_everything():
    client = QdrantClient(path=QDRANT_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    with open(BM25_FILE, "rb") as f:
        bm25_data = pickle.load(f)

    return client, embed_model, reranker, bm25_data["bm25"], bm25_data["chunks"]


def dense_search(client, model, query, top_k):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=top_k
    ).points
    return [r.payload["chunk_id"] for r in results]


def sparse_search(bm25, chunks, query, top_k):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk["chunk_id"] for chunk, score in ranked[:top_k]]


def reciprocal_rank_fusion(dense_ids, sparse_ids, k=RRF_K):
    scores = {}
    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(sparse_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, score in fused]


def rerank(query, candidate_chunks, reranker, top_n):
    # Cross-encoder needs (query, chunk_text) pairs to score together.
    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)

    scored = list(zip(candidate_chunks, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [(chunk, float(score)) for chunk, score in scored[:top_n]]


def hybrid_retrieve_and_rerank(query, client, embed_model, reranker, bm25, chunks, chunk_lookup):
    dense_ids = dense_search(client, embed_model, query, TOP_K_PER_METHOD)
    sparse_ids = sparse_search(bm25, chunks, query, TOP_K_PER_METHOD)
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)

    candidates = [chunk_lookup[cid] for cid in fused_ids[:CANDIDATES_FOR_RERANK]]
    reranked = rerank(query, candidates, reranker, FINAL_TOP_N)
    return reranked


if __name__ == "__main__":
    print("Loading indexes and models (reranker loads on first run, be patient)...")
    client, embed_model, reranker, bm25, chunks = load_everything()
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    print("Ready. Type a question (or 'quit' to exit).\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        results = hybrid_retrieve_and_rerank(
            query, client, embed_model, reranker, bm25, chunks, chunk_lookup
        )

        print(f"\nTop {len(results)} reranked chunks:\n")
        for i, (chunk, score) in enumerate(results, 1):
            preview = chunk["text"][:200].replace("\n", " ")
            print(f"[{i}] score={score:.3f} from '{chunk['source_doc']}' (chunk {chunk['chunk_index_in_doc']})")
            print(f"    {preview}...\n")
