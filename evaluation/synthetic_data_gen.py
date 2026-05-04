"""
synthetic_data_gen.py
---------------------
Generates synthetic student Q&A test cases from ingested course material.
Output: evaluation/data/synthetic_test_data.json
"""

import os
import sys
import json
import random
import chromadb
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_client import generate_json

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
_OUTPUT_FILE = os.path.join(_OUTPUT_DIR, "synthetic_test_data.json")

COURSES = [
    "deep_learning",
    "natural_language_processing",
    "big_data_analytics",
    "forecasting_methods",
    "text_and_image_analytics",
]

QUESTIONS_PER_COURSE = 10


def _get_sample_chunks(course_key: str, n: int = 40) -> list:
    """
    Retrieves a sample of chunks for a course.
    Uses query() with a neutral zero vector rather than get() with a
    where filter, to avoid ChromaDB version issues.
    Falls back to manual filtering via get() if query() fails.
    """
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    try:
        col = client.get_collection("chunks_overlap")
        dummy_embedding = [0.0] * 384

        results = col.query(
            query_embeddings=[dummy_embedding],
            n_results=min(n * 3, col.count()),
            where={
                "$and": [
                    {"course_key": {"$eq": course_key}},
                    {"unit_key": {"$ne": "unit_00"}}
                ]
            },
            include=["documents", "metadatas"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        items = [
            (doc, meta) for doc, meta in zip(docs, metas)
            if doc and len(doc) > 200
        ]
        random.shuffle(items)
        return items[:n]

    except Exception as e:
        print(f"  query() failed ({e}), trying fallback...")
        try:
            col = client.get_collection("chunks_overlap")
            results = col.get(include=["documents", "metadatas"])
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            items = [
                (doc, meta) for doc, meta in zip(docs, metas)
                if doc and len(doc) > 200
                and meta.get("course_key") == course_key
                and meta.get("unit_key") != "unit_00"
            ]
            random.shuffle(items)
            return items[:n]
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return []


def _generate_qa_pairs(chunk_text: str, course_title: str, unit_title: str) -> list:
    prompt = (
        f"You are creating a student exam question bank for the course '{course_title}', "
        f"unit on '{unit_title}'.\n\n"
        f"Based on the following course material, generate 2 realistic student "
        f"questions with correct answers.\n\n"
        f"Material:\n{chunk_text[:800]}\n\n"
        f"Return a JSON array with exactly 2 objects. Each object must have:\n"
        f'  "question": a clear student question about the material\n'
        f'  "ground_truth": a correct answer in 2-4 sentences\n\n'
        f"Return only the JSON array. No other text."
    )
    try:
        raw = generate_json(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "questions" in parsed:
            return parsed["questions"]
        return []
    except Exception as e:
        print(f"    Parse error: {e}")
        return []


def generate_synthetic_data():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    all_cases = []

    for course_key in COURSES:
        print(f"\nGenerating: {course_key}")
        chunks = _get_sample_chunks(course_key)
        print(f"  Retrieved {len(chunks)} eligible chunks.")

        if not chunks:
            print(f"  Skipping - no chunks found.")
            continue

        course_title = chunks[0][1].get("course_title", course_key)
        generated = 0
        idx = 0

        while generated < QUESTIONS_PER_COURSE and idx < len(chunks):
            doc, meta = chunks[idx]
            idx += 1
            unit_title = meta.get("unit_title", "")
            print(f"  Chunk {idx} (unit: {meta.get('unit_key','?')})...", end=" ", flush=True)

            pairs = _generate_qa_pairs(doc, course_title, unit_title)
            print(f"{len(pairs)} pairs.")

            for pair in pairs:
                if generated >= QUESTIONS_PER_COURSE:
                    break
                q = pair.get("question", "").strip()
                gt = pair.get("ground_truth", "").strip()
                if not q or not gt:
                    continue
                all_cases.append({
                    "question": q,
                    "ground_truth": gt,
                    "source_chunk": doc,
                    "course_key": course_key,
                    "course_title": course_title,
                    "unit_key": meta.get("unit_key", ""),
                    "unit_title": unit_title,
                    "source_file": meta.get("source_file", "")
                })
                generated += 1

        print(f"  Done: {generated} test cases.")

    with open(_OUTPUT_FILE, "w") as f:
        json.dump(all_cases, f, indent=2)

    print(f"\nTotal: {len(all_cases)} test cases -> {_OUTPUT_FILE}")
    return all_cases


if __name__ == "__main__":
    generate_synthetic_data()
