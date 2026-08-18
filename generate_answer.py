"""
Phase 5: generate answers with enforced, verified citations.

Takes your question, runs it through hybrid retrieval + reranking
(Phases 3-4), then asks Claude to answer using ONLY the retrieved
chunks, citing which chunk supports each claim. Finally, verifies
that every citation the model gives actually points to a real
retrieved chunk (a lightweight sanity check against hallucinated
sources).

Run with: python generate_answer.py
"""

import os
import json
import pickle

from dotenv import load_dotenv
from anthropic import Anthropic
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "ask_my_docs"
BM25_FILE = "bm25_index.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "claude-sonnet-5"

TOP_K_PER_METHOD = 10
RRF_K = 60
CANDIDATES_FOR_RERANK = 10
FINAL_TOP_N = 5

client_anthropic = Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically


def load_everything():
    client = QdrantClient(path=QDRANT_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    with open(BM25_FILE, "rb") as f:
        bm25_data = pickle.load(f)
    return client, embed_model, reranker, bm25_data["bm25"], bm25_data["chunks"]


def dense_search(client, model, query, top_k):
    query_vector = model.encode(query).tolist()
    results = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k).points
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
    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)
    scored = list(zip(candidate_chunks, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, score in scored[:top_n]]


def retrieve_top_chunks(query, client, embed_model, reranker, bm25, chunks, chunk_lookup):
    dense_ids = dense_search(client, embed_model, query, TOP_K_PER_METHOD)
    sparse_ids = sparse_search(bm25, chunks, query, TOP_K_PER_METHOD)
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)
    candidates = [chunk_lookup[cid] for cid in fused_ids[:CANDIDATES_FOR_RERANK]]
    return rerank(query, candidates, reranker, FINAL_TOP_N)


def build_prompt(query, top_chunks):
    # Number the chunks so the model can cite them by number.
    numbered_sources = "\n\n".join(
        f"[{i+1}] (from '{c['source_doc']}')\n{c['text']}"
        for i, c in enumerate(top_chunks)
    )

    return f"""You are answering a question using ONLY the numbered sources below.
Do not use any outside knowledge. If the sources don't contain the answer, say so clearly.

Every factual claim in your answer must be followed by a citation like [1] or [2],
referring to the source number it came from.

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "answer": "your answer text with inline citations like [1]",
  "citations": [
    {{"claim": "short phrase of the claim", "source_number": 1}}
  ]
}}

SOURCES:
{numbered_sources}

QUESTION: {query}
"""


def generate_answer(query, top_chunks):
    prompt = build_prompt(query, top_chunks)

    response = client_anthropic.messages.create(
        model=LLM_MODEL_NAME,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    # response.content can include non-text blocks (e.g. thinking blocks)
    # before the actual answer, so find the text block explicitly instead
    # of assuming it's first.
    text_blocks = [block.text for block in response.content if block.type == "text"]
    raw_text = "".join(text_blocks).strip()

    # Models sometimes wrap JSON in ```json fences despite instructions - strip if present.
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"answer": raw_text, "citations": [], "verified": False, "parse_error": True}

    return parsed


def verify_citations(parsed, num_sources):
    """Check every cited source_number actually exists among retrieved sources."""
    citations = parsed.get("citations", [])
    valid_numbers = set(range(1, num_sources + 1))

    all_valid = True
    for citation in citations:
        source_num = citation.get("source_number")
        if source_num not in valid_numbers:
            all_valid = False
            citation["valid"] = False
        else:
            citation["valid"] = True

    parsed["verified"] = all_valid
    return parsed


if __name__ == "__main__":
    print("Loading indexes and models...")
    client, embed_model, reranker, bm25, chunks = load_everything()
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    print("Ready. Type a question (or 'quit' to exit).\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        top_chunks = retrieve_top_chunks(query, client, embed_model, reranker, bm25, chunks, chunk_lookup)

        print("\nGenerating answer...\n")
        result = generate_answer(query, top_chunks)
        result = verify_citations(result, len(top_chunks))

        print("ANSWER:")
        print(result.get("answer", "(no answer returned)"))
        print(f"\nCitations verified: {result.get('verified')}")
        print("\nSources used:")
        for i, c in enumerate(top_chunks, 1):
            print(f"  [{i}] {c['source_doc']} (chunk {c['chunk_index_in_doc']})")
        print()
