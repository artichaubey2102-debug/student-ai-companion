"""
evaluate_chunking.py
--------------------
Compares three chunking strategies - fixed, overlap, semantic - using
the synthetic test dataset.

For each test question, retrieves the top-5 chunks from each strategy's
ChromaDB collection and computes:

  - Context Precision:  fraction of retrieved chunks that are relevant
  - Context Recall:     whether the ground-truth information was retrieved
  - Mean Chunk Length:  average character length of retrieved chunks
  - Retrieval Score:    cosine similarity of top-1 chunk to query

Results saved to evaluation/data/chunking_results.json and
evaluation/data/chunking_summary.csv
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
from utils.llm_client import generate_json

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_TEST_FILE = os.path.join(_DATA_DIR, "synthetic_test_data.json")
_RESULTS_FILE = os.path.join(_DATA_DIR, "chunking_results.json")
_SUMMARY_FILE = os.path.join(_DATA_DIR, "chunking_summary.csv")

STRATEGIES = ["fixed", "overlap", "semantic"]
TOP_K = 5


def _load_test_data() -> list:
    with open(_TEST_FILE) as f:
        return json.load(f)


def _retrieve_chunks(query: str, course_key: str, strategy: str, k: int = TOP_K) -> list:
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    try:
        col = client.get_collection(f"chunks_{strategy}")
        q_emb = embed_query(query)
        if not q_emb:
            return []
        results = col.query(
            query_embeddings=[q_emb],
            n_results=min(k, col.count()),
            where={"course_key": {"$eq": course_key}},
            include=["documents", "metadatas", "distances"]
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "metadata": meta,
                "score": round(1.0 - dist, 4)
            })
        return chunks
    except Exception as e:
        print(f"  Warning: retrieval failed for {strategy}: {e}")
        return []


def _assess_relevance(question: str, ground_truth: str, chunk_text: str) -> float:
    """
    Uses the LLM to judge whether a chunk is relevant to the question
    and contains information supporting the ground truth answer.
    Returns 1.0 (relevant) or 0.0 (not relevant).
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Ground truth answer: {ground_truth}\n\n"
        f"Retrieved chunk: {chunk_text[:500]}\n\n"
        f"Does this chunk contain information that is relevant to answering "
        f"the question and supports the ground truth? "
        f'Return JSON: {{"relevant": true}} or {{"relevant": false}}'
    )
    try:
        raw = generate_json(prompt)
        parsed = json.loads(raw)
        return 1.0 if parsed.get("relevant") else 0.0
    except Exception:
        # Fallback: keyword overlap heuristic
        gt_words = set(ground_truth.lower().split())
        chunk_words = set(chunk_text.lower().split())
        overlap = len(gt_words & chunk_words) / max(len(gt_words), 1)
        return 1.0 if overlap > 0.15 else 0.0


def evaluate_chunking():
    os.makedirs(_DATA_DIR, exist_ok=True)
    test_cases = _load_test_data()
    print(f"Loaded {len(test_cases)} test cases.")

    all_results = []

    for strategy in STRATEGIES:
        print(f"\nEvaluating strategy: {strategy}")
        strategy_results = []

        for i, case in enumerate(test_cases):
            print(f"  Case {i+1}/{len(test_cases)}: {case['question'][:60]}...")

            chunks = _retrieve_chunks(
                case["question"], case["course_key"], strategy
            )

            if not chunks:
                strategy_results.append({
                    "question": case["question"],
                    "strategy": strategy,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "top1_score": 0.0,
                    "mean_chunk_length": 0,
                    "chunks_retrieved": 0
                })
                continue

            relevance_scores = []
            for chunk in chunks:
                score = _assess_relevance(
                    case["question"],
                    case["ground_truth"],
                    chunk["text"]
                )
                relevance_scores.append(score)

            context_precision = (
                sum(relevance_scores) / len(relevance_scores)
                if relevance_scores else 0.0
            )
            context_recall = 1.0 if sum(relevance_scores) > 0 else 0.0
            top1_score = chunks[0]["score"] if chunks else 0.0
            mean_chunk_length = int(np.mean([len(c["text"]) for c in chunks]))

            result = {
                "question": case["question"],
                "course": case["course_title"],
                "unit": case["unit_title"],
                "strategy": strategy,
                "context_precision": round(context_precision, 4),
                "context_recall": round(context_recall, 4),
                "top1_score": round(top1_score, 4),
                "mean_chunk_length": mean_chunk_length,
                "chunks_retrieved": len(chunks)
            }
            strategy_results.append(result)

        all_results.extend(strategy_results)

        prec = np.mean([r["context_precision"] for r in strategy_results])
        rec = np.mean([r["context_recall"] for r in strategy_results])
        score = np.mean([r["top1_score"] for r in strategy_results])
        avg_len = np.mean([r["mean_chunk_length"] for r in strategy_results])
        print(f"  Precision: {prec:.3f} | Recall: {rec:.3f} | "
              f"Top1 Score: {score:.3f} | Avg Length: {avg_len:.0f}")

    with open(_RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(all_results)
    summary = df.groupby("strategy").agg(
        context_precision=("context_precision", "mean"),
        context_recall=("context_recall", "mean"),
        top1_score=("top1_score", "mean"),
        mean_chunk_length=("mean_chunk_length", "mean"),
        n_cases=("question", "count")
    ).round(4).reset_index()

    summary.to_csv(_SUMMARY_FILE, index=False)
    print(f"\nResults saved.")
    print(f"  Detailed: {_RESULTS_FILE}")
    print(f"  Summary:  {_SUMMARY_FILE}")
    print(f"\nChunking Strategy Summary:")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    evaluate_chunking()
