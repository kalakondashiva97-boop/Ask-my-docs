"""
Phase 6: evaluation pipeline.

Runs every question in golden_dataset.json through the full pipeline
(hybrid retrieval -> rerank -> generate -> verify citations) and scores:

  1. Retrieval accuracy: did the expected source document appear
     anywhere in the top retrieved/reranked chunks?
  2. Citation validity rate: were all citations the LLM gave actually
     real (pointing to retrieved chunks)?

Prints a scorecard and saves full results to eval_results.json.

Run with: python eval.py
"""

import json
import pickle

from dotenv import load_dotenv
from anthropic import Anthropic
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

GOLDEN_DATASET_FILE = "golden_dataset.json"
RESULTS_FILE = "eval_results.json"

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

# Minimum scores required to "pass" - tune these as your pipeline improves.
RETRIEVAL_ACCURACY_THRESHOLD = 0.75
CITATION_VALIDITY_THRESHOLD = 0.90

client_anthropic = Anthropic()


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
    numbered_sources = "\n\n".join(
        f"[{i+1}] (from '{c['source_doc']}')\n{c['text']}"
        for i, c in enumerate(top_chunks)
    )
    return f"""You are answering a question using ONLY the numbered sources below.
Do not use any outside knowledge. If the sources don't contain the answer, say so clearly.

Every factual claim in your answer must be followed by a citation like [1] or [2].

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
        model=LLM_MODEL_NAME, max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    raw_text = "".join(text_blocks).strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"answer": raw_text, "citations": [], "parse_error": True}


def evaluate_one(item, client, embed_model, reranker, bm25, chunks, chunk_lookup):
    question = item["question"]
    expected_doc = item["expected_source_doc"]

    top_chunks = retrieve_top_chunks(question, client, embed_model, reranker, bm25, chunks, chunk_lookup)

    retrieved_docs = {c["source_doc"] for c in top_chunks}
    retrieval_hit = expected_doc in retrieved_docs

    result = generate_answer(question, top_chunks)
    citations = result.get("citations", [])
    valid_numbers = set(range(1, len(top_chunks) + 1))
    valid_citations = [c for c in citations if c.get("source_number") in valid_numbers]
    citation_validity = (len(valid_citations) / len(citations)) if citations else 1.0

    return {
        "question": question,
        "expected_source_doc": expected_doc,
        "retrieved_docs": list(retrieved_docs),
        "retrieval_hit": retrieval_hit,
        "answer": result.get("answer", ""),
        "num_citations": len(citations),
        "citation_validity": citation_validity,
    }


if __name__ == "__main__":
    print("Loading indexes and models...")
    client, embed_model, reranker, bm25, chunks = load_everything()
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    with open(GOLDEN_DATASET_FILE, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    print(f"Running evaluation on {len(golden_dataset)} questions...\n")

    results = []
    for i, item in enumerate(golden_dataset, 1):
        print(f"[{i}/{len(golden_dataset)}] {item['question']}")
        result = evaluate_one(item, client, embed_model, reranker, bm25, chunks, chunk_lookup)
        results.append(result)
        status = "PASS" if result["retrieval_hit"] else "FAIL"
        print(f"    retrieval: {status} | citation validity: {result['citation_validity']:.0%}\n")

    # Aggregate scorecard
    retrieval_accuracy = sum(r["retrieval_hit"] for r in results) / len(results)
    avg_citation_validity = sum(r["citation_validity"] for r in results) / len(results)

    print("=" * 50)
    print("SCORECARD")
    print("=" * 50)
    print(f"Retrieval accuracy:    {retrieval_accuracy:.0%}  (threshold: {RETRIEVAL_ACCURACY_THRESHOLD:.0%})")
    print(f"Citation validity:     {avg_citation_validity:.0%}  (threshold: {CITATION_VALIDITY_THRESHOLD:.0%})")

    passed = (
        retrieval_accuracy >= RETRIEVAL_ACCURACY_THRESHOLD
        and avg_citation_validity >= CITATION_VALIDITY_THRESHOLD
    )
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "retrieval_accuracy": retrieval_accuracy,
            "citation_validity": avg_citation_validity,
            "passed": passed,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {RESULTS_FILE}")

    # Exit code 1 on failure - this is what lets CI (Phase 7) block a bad change.
    exit(0 if passed else 1)
