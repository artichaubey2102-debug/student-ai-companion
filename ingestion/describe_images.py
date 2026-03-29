"""
describe_images.py
------------------
Pre-processes images from one or multiple PDFs using the local Ollama
moondream model and saves descriptions to per-document cache files.

Can be run in three modes:

1. Single PDF:
    python3 -m ingestion.describe_images \
        --pdf data/pdfs/goodfellow.pdf \
        --course deep_learning

2. All PDFs in a folder mapped to courses via a config file:
    python3 -m ingestion.describe_images \
        --all \
        --config data/pdf_config.json

3. Check cache status across all configured PDFs:
    python3 -m ingestion.describe_images \
        --all \
        --config data/pdf_config.json \
        --status-only

pdf_config.json format:
[
    {
        "pdf": "data/pdfs/goodfellow.pdf",
        "course": "deep_learning"
    },
    {
        "pdf": "data/pdfs/jurafsky.pdf",
        "course": "natural_language_processing"
    }
]
"""

import argparse
import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.pdf_extractor import extract_from_pdf, get_pdf_metadata
from ingestion.vision_describer import (
    describe_pdf_images_to_cache,
    get_cache_stats
)
from utils.syllabus_lookup import load_syllabus


def process_single_pdf(pdf_path: str, course_key: str, syllabus: dict):
    """
    Runs image description for one PDF and saves to cache.
    Resumes from cache if previously interrupted.
    """
    if not os.path.exists(pdf_path):
        print(f"  ERROR: File not found: {pdf_path}")
        return False

    if course_key not in syllabus:
        print(f"  ERROR: '{course_key}' not in syllabus.json.")
        print(f"  Valid keys: {list(syllabus.keys())}")
        return False

    course_title = syllabus[course_key]["title"]
    filename = os.path.basename(pdf_path)

    print(f"\n{'='*60}")
    print(f"PDF      : {filename}")
    print(f"Course   : {course_title}")
    print(f"{'='*60}")

    print("Extracting pages...")
    pages = extract_from_pdf(pdf_path)
    print(f"{len(pages)} pages extracted.")

    describe_pdf_images_to_cache(
        pages=pages,
        course_key=course_key,
        course_title=course_title,
        unit_title="",
        filename=filename
    )
    return True


def print_status(pdf_path: str, course_key: str, syllabus: dict):
    filename = os.path.basename(pdf_path)
    course_title = syllabus.get(course_key, {}).get("title", course_key)
    stats = get_cache_stats(course_key, filename)

    if stats["exists"]:
        print(
            f"  {filename[:45]:<45} "
            f"{course_title[:25]:<25} "
            f"{stats['descriptions_saved']} descriptions cached"
        )
    else:
        print(
            f"  {filename[:45]:<45} "
            f"{course_title[:25]:<25} "
            f"No cache yet"
        )


def load_config(config_path: str) -> list[dict]:
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating a template config file for you...")
        template = [
            {
                "pdf": "data/pdfs/your_first_book.pdf",
                "course": "deep_learning"
            },
            {
                "pdf": "data/pdfs/your_second_book.pdf",
                "course": "natural_language_processing"
            }
        ]
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(template, f, indent=2)
        print(f"Template written to {config_path}")
        print("Edit it with your actual PDF paths and course keys, then re-run.")
        sys.exit(0)

    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process PDF images using local Ollama moondream model."
    )
    parser.add_argument(
        "--pdf",
        help="Path to a single PDF file."
    )
    parser.add_argument(
        "--course",
        help="Course key for the single PDF, e.g. 'deep_learning'."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all PDFs defined in the config file."
    )
    parser.add_argument(
        "--config",
        default="data/pdf_config.json",
        help="Path to the JSON config file listing all PDFs and courses. "
             "Default: data/pdf_config.json"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print cache status for all PDFs and exit without processing."
    )
    args = parser.parse_args()

    syllabus = load_syllabus()

    # Single PDF mode
    if args.pdf and args.course:
        if args.status_only:
            print_status(args.pdf, args.course, syllabus)
        else:
            process_single_pdf(args.pdf, args.course, syllabus)
        return

    # Batch mode using config file
    if args.all:
        entries = load_config(args.config)

        if args.status_only:
            print(f"\nCache status for {len(entries)} configured PDFs:\n")
            print(
                f"  {'File':<45} {'Course':<25} Status"
            )
            print(f"  {'-'*45} {'-'*25} {'-'*25}")
            for entry in entries:
                print_status(entry["pdf"], entry["course"], syllabus)
            print()
            return

        total = len(entries)
        print(f"\nBatch image description: {total} PDF(s) to process.")
        print("Ollama moondream model will be used for all image descriptions.")
        print("Progress is saved after every image. Safe to interrupt and resume.\n")

        for i, entry in enumerate(entries, 1):
            print(f"\nProcessing PDF {i} of {total}...")
            process_single_pdf(entry["pdf"], entry["course"], syllabus)

        print(f"\n{'='*60}")
        print("All PDFs processed. Cache status summary:")
        print()
        for entry in entries:
            print_status(entry["pdf"], entry["course"], syllabus)
        print()
        print("You can now run the ingestion pipeline.")
        print("  python3 -m ingestion.ingest_all --config data/pdf_config.json")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
