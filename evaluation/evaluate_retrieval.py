"""
evaluate_retrieval.py
---------------------
Compares cosine similarity vs MMR retrieval using the synthetic test dataset.

Metrics computed per method:
  - Precision@K:  fraction of top-K retrieved chunks that are relevant
  - Recall@K:     fraction of relevant chunks found in top-K
  - MRR:          Mean Reciprocal Rank of first relevant chunk
  - NDCG:         Normalised Discounted Cumulative Gain
  - Diversity:    average pairwise dissimilarity among retrieved chunks

Results saved to evaluation/data/retrieval_results.json and
evaluation/data/retrieval_summary.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import chromadb
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.embedder import embed_query
from rag.retriever import retrieve_cosine, retrieve_mmr
from utils.llm_client import generate_json

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_TEST_FILE = os.path.join(_DATA_DIR, "synthetic_test_data.json")
_RESULTS_FILE = os.path.join(_DATA_DIR, "retrieval_results.json")
_SUMMARY_FILE = os.path.join(_DATA_DIR, "retrieval_summary.csv")

TOP_K = 5


def _load_test_data() -> list:
    with open(_TEST_FILE) as f:
        return json.load(f)


def _assess_relevance(question: str, ground_truth: str, chunk_text: str) -> float:
    prompt = (
        f"Question: {question}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        f"Chunk: {chunk_text[:500]}\n\n"
        f"Is this chunk relevant to answering the question? "
        f'Return JSON: {{"relevant": true}} or {{"relevant": false}}'
    )
    try:
        raw = generate_json(prompt)
        parsed = json.loads(raw)
        return 1.0 if parsed.get("relevant") else 0.0
    except Exception:
        gt_words = set(ground_truth.lower().split())
        chunk_words = set(chunk_text.lower().split())
        overlap = len(gt_words & chunk_words) / max(len(gt_words), 1)
        return 1.0 if overlap > 0.15 else 0.0


def _compute_ndcg(relevance_scores: list, k: int) -> float:
    dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(relevance_scores[:k])
    )
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0.0


def _compute_mrr(relevance_scores: list) -> float:
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def _compute_diversity(chunks: list) -> float:
    """
    Average pairwise cosine dissimilarity among retrieved chunks.
    Higher diversity means less redundant retrieval.
    """
    if len(chunks) < 2:
        return 0.0
    embeddings = [embed_query(c["text"]) for c in chunks]
    embeddings = [e for e in embeddings if e]
    if len(embeddings) < 2:
        return 0.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a, b = np.array(embeddings[i]), np.array(embeddings[j])
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm > 0:
                sims.append(float(np.dot(a, b) / norm))
    return round(1.0 - np.mean(sims), 4) if sims else 0.0


def _build_filter(course_key: str) -> dict:
    return {"course_key": {"$eq": course_key}}


def evaluate_retrieval():
    os.makedirs(_DATA_DIR, exist_ok=True)
    test_cases = _load_test_data()
    print(f"Loaded {len(test_cases)} test cases.")

    all_results = []

    for method_name in ["cosine", "mmr"]:
        print(f"\nEvaluating method: {method_name}")

        for i, case in enumerate(test_cases):
            print(f"  Case {i+1}/{len(test_cases)}: {case['question'][:55]}...")

            chroma_filter = _build_filter(case["course_key"])

            if method_name == "cosine":
                chunks = retrieve_cosine(
                    case["question"], chroma_filter, top_k=TOP_K
                )
            else:
                chunks = retrieve_mmr(
                    case["question"], chroma_filter, top_k=TOP_K
                )

            if not chunks:
                all_results.append({
                    "question": case["question"],
                    "course": case["course_title"],
                    "method": method_name,
                    "precision_at_k": 0.0,
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "ndcg": 0.0,
                    "diversity": 0.0,
                    "chunks_retrieved": 0
                })
                continue

            relevance = [
                _assess_relevance(
                    case["question"], case["ground_truth"], c["text"]
                )
                for c in chunks
            ]

            precision = sum(relevance) / len(relevance)
            recall = 1.0 if sum(relevance) > 0 else 0.0
            mrr = _compute_mrr(relevance)
            ndcg = _compute_ndcg(relevance, TOP_K)
            diversity = _compute_diversity(chunks)

            all_results.append({
                "question": case["question"],
                "course": case["course_title"],
                "method": method_name,
                "precision_at_k": round(precision, 4),
                "recall_at_k": round(recall, 4),
                "mrr": round(mrr, 4),
                "ndcg": round(ndcg, 4),
                "diversity": round(diversity, 4),
                "chunks_retrieved": len(chunks)
            })

        method_results = [r for r in all_results if r["method"] == method_name]
        print(f"  Precision@{TOP_K}: {np.mean([r['precision_at_k'] for r in method_results]):.3f}")
        print(f"  MRR:              {np.mean([r['mrr'] for r in method_results]):.3f}")
        print(f"  NDCG:             {np.mean([r['ndcg'] for r in method_results]):.3f}")
        print(f"  Diversity:        {np.mean([r['diversity'] for r in method_results]):.3f}")

    with open(_RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(all_results)
    summary = df.groupby("method").agg(
        precision_at_k=("precision_at_k", "mean"),
        recall_at_k=("recall_at_k", "mean"),
        mrr=("mrr", "mean"),
        ndcg=("ndcg", "mean"),
        diversity=("diversity", "mean"),
        n_cases=("question", "count")
    ).round(4).reset_index()

    summary.to_csv(_SUMMARY_FILE, index=False)
    print(f"\nResults saved.")
    print(f"  Detailed: {_RESULTS_FILE}")
    print(f"  Summary:  {_SUMMARY_FILE}")
    print(f"\nRetrieval Method Summary:")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    evaluate_retrieval()
