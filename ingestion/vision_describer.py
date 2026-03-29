"""
vision_describer.py
-------------------
Describes images found in PDF pages using a local vision model via Ollama.

Uses moondream2 running locally through Ollama. This requires no API key,
has no rate limits, no quota, and no internet connection after the
one-time model download.

Setup (one time only):
    1. Download Ollama from https://ollama.com and install it.
    2. Run: ollama pull moondream
    3. Run: pip install ollama

The description process saves progress to a JSON cache file on disk after
every image. If interrupted, re-running the same command resumes exactly
where it left off without re-processing any image already described.
"""

import os
import io
import json
import hashlib
import base64
from PIL import Image

_MIN_IMAGE_AREA = 8000
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "image_cache")
_OLLAMA_MODEL = "moondream"


def _image_hash(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return hashlib.md5(buffer.getvalue()).hexdigest()


def _image_to_base64(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, cache_path: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def _is_worth_describing(pil_image: Image.Image) -> bool:
    w, h = pil_image.size
    return (w * h) >= _MIN_IMAGE_AREA


def _describe_single_image(
    pil_image: Image.Image,
    course_title: str,
    unit_title: str
) -> str:
    """
    Calls the local Ollama moondream model to describe one image.
    No internet connection or API key required.
    Returns a description string or empty string on failure.
    """
    try:
        import ollama

        prompt = (
            f"This image is from a university course titled '{course_title}'. "
            "Describe what this figure, diagram, chart, or illustration shows. "
            "Focus on the technical content. "
            "Write a clear factual description of two to four sentences "
            "that would help a student understand what the image conveys."
        )

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        response = ollama.chat(
            model=_OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes]
                }
            ]
        )

        return response["message"]["content"].strip()

    except Exception as e:
        print(f"  Warning: vision call failed. Reason: {e}")
        print("  Check that Ollama is running and moondream is installed.")
        print("  Run: ollama pull moondream")
        return ""


def describe_pdf_images_to_cache(
    pages: list[dict],
    course_key: str,
    course_title: str,
    unit_title: str,
    filename: str
) -> str:
    """
    Processes all images across all pages and saves descriptions to a
    per-document JSON cache file. Resumes safely if interrupted.

    Args:
        pages:        List of page dicts from pdf_extractor.
        course_key:   Used to name the cache file.
        course_title: Passed to the vision prompt.
        unit_title:   Passed to the vision prompt.
        filename:     Source PDF filename, used to name the cache file.

    Returns:
        Path to the cache file.
    """
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    cache_path = os.path.join(_CACHE_DIR, f"{course_key}_{safe_name}.json")
    cache = _load_cache(cache_path)

    total_images = sum(len(p["images"]) for p in pages)
    skipped_small = 0
    skipped_cached = 0
    described = 0
    failed = 0

    print(f"  Image description via local Ollama ({_OLLAMA_MODEL})")
    print(f"  Total images in document: {total_images}")
    print(f"  Cache file: {cache_path}")
    print(f"  Already cached: {len(cache)} images\n")

    for page in pages:
        for img_data in page["images"]:
            pil_img = img_data.get("image")
            if pil_img is None:
                continue

            if not _is_worth_describing(pil_img):
                skipped_small += 1
                continue

            img_key = _image_hash(pil_img)

            if img_key in cache:
                skipped_cached += 1
                continue

            description = _describe_single_image(
                pil_img, course_title, unit_title
            )

            if description:
                cache[img_key] = description
                described += 1
                _save_cache(cache, cache_path)
                print(
                    f"  [{described} new] Page {page['page_number']}: "
                    f"{description[:80]}..."
                )
            else:
                failed += 1

    print(
        f"\n  Complete.\n"
        f"  Described this run: {described}\n"
        f"  From cache: {skipped_cached}\n"
        f"  Too small (skipped): {skipped_small}\n"
        f"  Failed: {failed}\n"
        f"  Total in cache: {len(cache)}"
    )

    return cache_path


def load_descriptions_from_cache(
    pages: list[dict],
    course_key: str,
    filename: str
) -> list[str]:
    """
    Reads pre-computed image descriptions from the cache file and
    returns them as an ordered list of strings.

    Called by ingest.py during content merging. If no cache exists,
    returns an empty list and prints a note.
    """
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    cache_path = os.path.join(_CACHE_DIR, f"{course_key}_{safe_name}.json")

    if not os.path.exists(cache_path):
        print(
            f"  Note: No image cache found for '{filename}'. "
            f"Run describe_images.py first to generate descriptions. "
            f"Proceeding with text and tables only."
        )
        return []

    cache = _load_cache(cache_path)
    descriptions = []

    for page in pages:
        for img_data in page.get("images", []):
            pil_img = img_data.get("image")
            if pil_img is None:
                continue
            if not _is_worth_describing(pil_img):
                continue
            img_key = _image_hash(pil_img)
            if img_key in cache and cache[img_key]:
                descriptions.append(cache[img_key])

    return descriptions


def get_cache_path(course_key: str, filename: str) -> str:
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    return os.path.join(_CACHE_DIR, f"{course_key}_{safe_name}.json")


def get_cache_stats(course_key: str, filename: str) -> dict:
    cache_path = get_cache_path(course_key, filename)
    if not os.path.exists(cache_path):
        return {"exists": False, "descriptions_saved": 0}
    cache = _load_cache(cache_path)
    return {
        "exists": True,
        "cache_path": cache_path,
        "descriptions_saved": len(cache)
    }
