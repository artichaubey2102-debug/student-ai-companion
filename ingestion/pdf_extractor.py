"""
pdf_extractor.py
----------------
Extracts text, tables, images, and chapter/section headings from PDFs.

Headings are detected by font size and cleaned to handle concatenated
text (e.g. "MaxPoolingLayer" -> "Max Pooling Layer") which is common
in PDFs where heading characters have no space glyphs between words.
"""

import pdfplumber
import os
import re
from PIL import Image
import io
from collections import Counter


def _clamp_bbox(bbox, page_bbox):
    px0, py0, px1, py1 = page_bbox
    x0, y0, x1, y1 = bbox
    return (
        max(x0, px0),
        max(y0, py0),
        min(x1, px1),
        min(y1, py1)
    )


def _split_concatenated_heading(text: str) -> str:
    """
    Splits concatenated heading text into readable words.
    Handles camelCase, section numbering, hyphenated line breaks,
    and special ligature characters common in PDF text extraction.

    Examples:
        "6.5Back-PropagationandOtherAlgorithms" -> "Back Propagation and Other Algorithms"
        "MaxPoolingLayer"                        -> "Max Pooling Layer"
        "ReviewQuestions"                        -> "Review Questions"
    """
    # Replace common ligature characters produced by PDF extraction
    text = text.replace("\ufb00", "ff").replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl").replace("\u2010", " ")
    text = text.replace("\u2011", " ").replace("\ufb03", "ffi")

    # Remove section numbering at the start (e.g. 6.5, 13.1.2)
    text = re.sub(r"^\d+(\.\d+)*\s*", "", text)

    # Remove embedded section numbers
    text = re.sub(r"\d+(\.\d+)+", " ", text)

    # Remove hyphenated line breaks (e.g. "Algo-rithms" -> "Algorithms")
    text = re.sub(r"([a-z])-([a-z])", r"\1\2", text)

    # Insert space before common lowercase connectors
    for word in ["and", "of", "the", "in", "for", "or", "with", "by"]:
        text = re.sub(rf"([a-z])({word})([A-Z])", r"\1 \2 \3", text)

    # Split camelCase: "MaxPooling" -> "Max Pooling"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Split acronym sequences: "HTMLParser" -> "HTML Parser"
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

    # Remove remaining hyphens used as separators
    text = re.sub(r"-", " ", text)

    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _extract_headings(page) -> str:
    """
    Detects chapter and section headings on a page by finding text
    rendered at a larger font size than the body text.
    Returns cleaned heading text or empty string.
    """
    try:
        chars = page.chars
        if not chars:
            return ""

        sizes = [round(c["size"]) for c in chars if c.get("size")]
        if not sizes:
            return ""

        size_counts = Counter(sizes)
        body_size = size_counts.most_common(1)[0][0]

        heading_chars = [
            c for c in chars
            if c.get("size") and round(c["size"]) > body_size + 1
        ]

        if not heading_chars:
            return ""

        heading_chars.sort(key=lambda c: (round(c["top"]), c["x0"]))

        words = []
        current_word = ""
        prev_x1 = None

        for c in heading_chars:
            text = c.get("text", "").strip()
            if not text:
                continue
            if prev_x1 is not None and c["x0"] - prev_x1 > 8:
                if current_word:
                    words.append(current_word)
                current_word = text
            else:
                current_word += text
            prev_x1 = c.get("x1", c["x0"])

        if current_word:
            words.append(current_word)

        raw_heading = " ".join(words).strip()
        return _split_concatenated_heading(raw_heading)

    except Exception:
        return ""


def extract_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts content from a PDF page by page.

    Returns a list of page dicts containing:
        page_number  (int)
        text         (str)   plain body text
        tables       (list)  each table as list of row lists
        images       (list)  each image as PIL Image with position
        heading      (str)   cleaned chapter or section heading
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_data = {
                "page_number": i + 1,
                "text": "",
                "tables": [],
                "images": [],
                "heading": ""
            }

            page_data["heading"] = _extract_headings(page)

            tables_found = page.find_tables()
            table_bboxes = []
            for t in tables_found:
                clamped = _clamp_bbox(t.bbox, page.bbox)
                if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
                    table_bboxes.append(clamped)

            for table in tables_found:
                rows = table.extract()
                if rows:
                    page_data["tables"].append(rows)

            try:
                if table_bboxes:
                    filtered_page = page
                    for bbox in table_bboxes:
                        filtered_page = filtered_page.outside_bbox(bbox)
                    raw_text = filtered_page.extract_text()
                else:
                    raw_text = page.extract_text()
            except Exception:
                raw_text = page.extract_text()

            page_data["text"] = raw_text.strip() if raw_text else ""

            for img_meta in page.images:
                try:
                    x0 = img_meta["x0"]
                    y0 = img_meta["y0"]
                    x1 = img_meta["x1"]
                    y1 = img_meta["y1"]
                    cx0, cy0, cx1, cy1 = _clamp_bbox(
                        (x0, y0, x1, y1), page.bbox
                    )
                    if cx1 <= cx0 or cy1 <= cy0:
                        continue
                    cropped = page.crop((cx0, cy0, cx1, cy1))
                    img_bytes = cropped.to_image(resolution=150).original
                    pil_img = img_bytes.convert("RGB")
                    page_data["images"].append({
                        "image": pil_img,
                        "bbox": (cx0, cy0, cx1, cy1),
                        "page_number": i + 1
                    })
                except Exception:
                    continue

            pages.append(page_data)

    return pages


def get_pdf_metadata(pdf_path: str) -> dict:
    with pdfplumber.open(pdf_path) as pdf:
        info = pdf.metadata or {}
        return {
            "filename": os.path.basename(pdf_path),
            "total_pages": len(pdf.pages),
            "title": info.get("Title", ""),
            "author": info.get("Author", "")
        }
