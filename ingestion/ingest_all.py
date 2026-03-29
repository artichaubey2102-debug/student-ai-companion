"""
ingest_all.py
-------------
Runs the full ingestion pipeline for all PDFs in the config file.

Supports resuming: if a PDF has already been ingested with a given
strategy, it is skipped automatically. Delete chroma_db/ to start fresh.

Usage:
    python3 -m ingestion.ingest_all --config data/pdf_config.json
    python3 -m ingestion.ingest_all --config data/pdf_config.json --all-strategies
    python3 -m ingestion.ingest_all --config data/pdf_config.json --status
"""

import argparse
import os
import sys
import json
import chromadb
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.ingest import ingest_pdf, ingest_for_all_strategies
from utils.syllabus_lookup import load_syllabus

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def load_config(config_path: str) -> list[dict]:
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)


def get_ingested_files(strategy: str) -> set[str]:
    """
    Returns the set of filenames already stored in a ChromaDB collection.
    Used to skip files that have already been ingested.
    """
    try:
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        collection = client.get_collection(f"chunks_{strategy}")
        results = collection.get(include=["metadatas"])
        filenames = set()
        for meta in results["metadatas"]:
            if meta and "source_file" in meta:
                filenames.add(meta["source_file"])
        return filenames
    except Exception:
        return set()


def print_status(entries: list[dict], syllabus: dict):
    strategies = ["fixed", "overlap", "semantic"]
    print(f"\n{'File':<40} {'Course':<25} {'fixed':>7} {'overlap':>9} {'semantic':>9}")
    print(f"{'-'*40} {'-'*25} {'-'*7} {'-'*9} {'-'*9}")

    for entry in entries:
        filename = os.path.basename(entry["pdf"])
        course_title = syllabus.get(
            entry["course"], {}
        ).get("title", entry["course"])[:24]

        counts = []
        for strategy in strategies:
            try:
                client = chromadb.PersistentClient(path=_CHROMA_PATH)
                col = client.get_collection(f"chunks_{strategy}")
                results = col.get(
                    where={"source_file": {"$eq": filename}},
                    include=["metadatas"]
                )
                counts.append(str(len(results["ids"])))
            except Exception:
                counts.append("0")

        print(
            f"{filename[:40]:<40} {course_title:<25} "
            f"{counts[0]:>7} {counts[1]:>9} {counts[2]:>9}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run ingestion pipeline for all configured PDFs."
    )
    parser.add_argument(
        "--config",
        default="data/pdf_config.json",
        help="Path to the JSON config file. Default: data/pdf_config.json"
    )
    parser.add_argument(
        "--strategy",
        default="overlap",
        choices=["fixed", "overlap", "semantic"],
        help="Chunking strategy. Default: overlap"
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run ingestion with all three chunking strategies."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show how many chunks are stored per file per strategy and exit."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if already in ChromaDB."
    )
    args = parser.parse_args()

    syllabus = load_syllabus()
    entries = load_config(args.config)

    if args.status:
        print_status(entries, syllabus)
        return

    strategies = (
        ["fixed", "overlap", "semantic"] if args.all_strategies
        else [args.strategy]
    )

    total = len(entries)
    print(f"\nBatch ingestion: {total} PDF(s), strategies: {strategies}")
    print("Embeddings: local sentence-transformers (no API quota)")
    if not args.force:
        print("Resume mode: files already ingested will be skipped.")
        print("Use --force to re-ingest everything.\n")

    all_summaries = []

    for i, entry in enumerate(entries, 1):
        pdf_path = entry["pdf"]
        course_key = entry["course"]
        filename = os.path.basename(pdf_path)

        if not os.path.exists(pdf_path):
            print(f"\nSkipping ({i}/{total}): File not found: {pdf_path}")
            continue

        if course_key not in syllabus:
            print(f"\nSkipping ({i}/{total}): Unknown course '{course_key}'")
            continue

        print(f"\n{'='*60}")
        print(f"PDF {i}/{total}: {filename}")
        print(f"Course: {syllabus[course_key]['title']}")

        for strategy in strategies:
            # Check if already ingested
            if not args.force:
                already_done = get_ingested_files(strategy)
                if filename in already_done:
                    print(
                        f"  Strategy '{strategy}': already ingested, skipping. "
                        f"Use --force to re-ingest."
                    )
                    continue

            summary = ingest_pdf(
                pdf_path,
                course_key,
                strategy=strategy,
                syllabus=syllabus
            )
            all_summaries.append(summary)

    # Final summary
    print(f"\n{'='*60}")
    print("Ingestion complete.\n")
    if all_summaries:
        print(
            f"{'File':<35} {'Course':<20} {'Strategy':<10} {'Chunks':>6}"
        )
        print(f"{'-'*35} {'-'*20} {'-'*10} {'-'*6}")
        for s in all_summaries:
            if s.get("status") == "success":
                print(
                    f"{s['filename'][:35]:<35} "
                    f"{s['course'][:20]:<20} "
                    f"{s['strategy']:<10} "
                    f"{s['chunks_stored']:>6}"
                )
            else:
                print(
                    f"{s.get('filename','unknown')[:35]:<35} "
                    f"FAILED: {s.get('reason','unknown')}"
                )

    print("\nRun with --status to see full chunk counts per file.")


if __name__ == "__main__":
    main()
