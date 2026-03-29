"""
ingest.py
---------
Orchestrates the full document ingestion pipeline.

Each chunk now carries the heading detected on its source page.
The heading is passed to the topic resolver as a high-weight signal
for syllabus unit matching, and also stored in chunk metadata so
it is visible during retrieval for debugging and evaluation.
"""

import os
import uuid
import chromadb
from dotenv import load_dotenv

from ingestion.pdf_extractor import extract_from_pdf, get_pdf_metadata
from ingestion.table_converter import tables_to_text_blocks
from ingestion.vision_describer import load_descriptions_from_cache
from ingestion.chunkers import chunk_text
from ingestion.topic_resolver import resolve_unit
from ingestion.embedder import embed_batch
from utils.syllabus_lookup import load_syllabus

load_dotenv()

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def _get_chroma_collection(strategy: str = "overlap") -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection_name = f"chunks_{strategy}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def _build_page_heading_map(pages: list[dict]) -> dict[int, str]:
    """
    Builds a map of page_number -> heading text.
    When a page has no heading, we carry forward the last seen heading
    so that body pages within a chapter inherit the chapter title.
    """
    heading_map = {}
    last_heading = ""
    for page in pages:
        h = page.get("heading", "").strip()
        if h:
            last_heading = h
        heading_map[page["page_number"]] = last_heading
    return heading_map


def _build_merged_content(
    pages: list[dict],
    course_key: str,
    course_title: str,
    filename: str
) -> list[dict]:
    """
    Merges text, tables, and cached image descriptions from all pages.

    Returns a list of content block dicts, each with:
        text         (str) the merged content
        page_number  (int) source page
        heading      (str) chapter/section heading for this page
    """
    image_descriptions = load_descriptions_from_cache(
        pages, course_key, filename
    )
    description_iter = iter(image_descriptions)

    blocks = []

    for page in pages:
        parts = []

        if page["text"]:
            parts.append(page["text"])

        table_blocks = tables_to_text_blocks(page["tables"])
        for block in table_blocks:
            parts.append(
                f"[Table from page {page['page_number']}]\n{block}"
            )

        for _ in page.get("images", []):
            desc = next(description_iter, None)
            if desc:
                parts.append(
                    f"[Figure from page {page['page_number']}] {desc}"
                )

        if parts:
            blocks.append({
                "text": "\n\n".join(parts),
                "page_number": page["page_number"],
                "heading": page.get("heading", "")
            })

    return blocks


def ingest_pdf(
    pdf_path: str,
    course_key: str,
    strategy: str = "overlap",
    syllabus: dict = None
) -> dict:
    """
    Runs the full ingestion pipeline for a single PDF.

    Each chunk is matched to a syllabus unit using both its body text
    and its source page heading. The heading is also stored in metadata
    so retrieval results can show which chapter a chunk came from.

    Args:
        pdf_path:   Path to the PDF file.
        course_key: Key matching syllabus.json, e.g. 'deep_learning'.
        strategy:   Chunking strategy: 'fixed', 'overlap', or 'semantic'.
        syllabus:   Pre-loaded syllabus dict. Loaded if None.

    Returns:
        Summary dict with ingestion statistics.
    """
    if syllabus is None:
        syllabus = load_syllabus()

    course_title = syllabus[course_key]["title"]
    pdf_meta = get_pdf_metadata(pdf_path)
    filename = pdf_meta["filename"]

    print(f"\nIngesting: {filename}")
    print(f"Course: {course_title} | Strategy: {strategy}")

    # Stage 1: Extract
    print("\nStage 1/5: Extracting content from PDF...")
    pages = extract_from_pdf(pdf_path)
    print(f"          {len(pages)} pages extracted.")

    # Stage 2: Merge content blocks with heading info preserved
    print("\nStage 2/5: Merging content...")
    content_blocks = _build_merged_content(
        pages, course_key, course_title, filename
    )

    # Build heading map that carries headings forward across pages
    heading_map = _build_page_heading_map(pages)

    merged_text = "\n\n".join(b["text"] for b in content_blocks)
    print(f"          {len(merged_text):,} characters.")

    if not merged_text.strip():
        print("  Warning: No usable text extracted. Aborting.")
        return {"status": "failed", "reason": "no text extracted",
                "chunks_stored": 0}

    # Stage 3: Chunk
    print(f"\nStage 3/5: Chunking with strategy '{strategy}'...")
    chunks = chunk_text(merged_text, strategy=strategy)
    print(f"          {len(chunks)} chunks produced.")

    if not chunks:
        return {"status": "failed", "reason": "no chunks produced",
                "chunks_stored": 0}

    # Build a heading for each chunk by matching chunk position
    # to the page heading map. We use a simple approach: assign
    # each chunk the heading of the page whose content it most
    # likely came from, estimated by character position.
    cumulative_lengths = []
    cumulative = 0
    for block in content_blocks:
        cumulative += len(block["text"])
        cumulative_lengths.append((cumulative, block["page_number"]))

    total_length = len(merged_text)

    def get_heading_for_chunk(chunk_text_local: str) -> str:
        pos = merged_text.find(chunk_text_local[:80])
        if pos == -1:
            return heading_map.get(1, "")
        for cum_len, page_num in cumulative_lengths:
            if pos <= cum_len:
                return heading_map.get(page_num, "")
        return heading_map.get(cumulative_lengths[-1][1], "")

    # Stage 4: Resolve units using body text + heading
    print("\nStage 4/5: Resolving syllabus units...")
    resolved = []
    unmatched_count = 0

    for chunk in chunks:
        heading = get_heading_for_chunk(chunk)
        unit_info = resolve_unit(chunk, syllabus, course_key, heading=heading)
        resolved.append((unit_info, heading))
        if unit_info["unit_key"] == "unit_00":
            unmatched_count += 1

    matched = len(chunks) - unmatched_count
    print(
        f"          Matched: {matched}/{len(chunks)} chunks "
        f"({100*matched//len(chunks)}%) | "
        f"unit_00: {unmatched_count} chunks"
    )

    # Stage 5: Embed
    print("\nStage 5/5: Embedding chunks...")
    embeddings = embed_batch(chunks)

    # Store in ChromaDB
    print("\nStoring in ChromaDB...")
    collection = _get_chroma_collection(strategy)

    ids, documents, metadatas, valid_embeddings = [], [], [], []

    for i, (chunk, (unit_info, heading), embedding) in enumerate(
        zip(chunks, resolved, embeddings)
    ):
        if not embedding:
            continue

        metadata = {
            "source_file": filename,
            "course_key": course_key,
            "course_title": course_title,
            "unit_key": unit_info["unit_key"],
            "unit_title": unit_info["unit_title"],
            "resolution_method": unit_info["resolution_method"],
            "heading": heading[:200] if heading else "",
            "chunk_index": i,
            "chunking_strategy": strategy,
            "page_count": pdf_meta["total_pages"]
        }

        ids.append(str(uuid.uuid4()))
        documents.append(chunk)
        metadatas.append(metadata)
        valid_embeddings.append(embedding)

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=valid_embeddings
        )

    summary = {
        "status": "success",
        "filename": filename,
        "course": course_title,
        "strategy": strategy,
        "pages_processed": len(pages),
        "chunks_stored": len(ids),
        "chunks_skipped": len(chunks) - len(ids),
        "unmatched_unit_00": unmatched_count
    }

    print(f"\nDone. {len(ids)} chunks stored in 'chunks_{strategy}'.")
    return summary


def ingest_for_all_strategies(pdf_path: str, course_key: str) -> list[dict]:
    """Runs ingestion with all three chunking strategies."""
    syllabus = load_syllabus()
    summaries = []
    for strategy in ["fixed", "overlap", "semantic"]:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print('='*60)
        summary = ingest_pdf(
            pdf_path, course_key,
            strategy=strategy,
            syllabus=syllabus
        )
        summaries.append(summary)
    return summaries
