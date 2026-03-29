"""
query_detector.py
-----------------
Detects course, unit, and topic from a student query using the local
Ollama LLM.

When a specific unit is detected, the ChromaDB filter includes both
that unit AND unit_00 (the catch-all for unclassified chunks) so that
no relevant content is missed due to ingestion-time classification gaps.

When no unit is detected, the filter scopes to the whole course or
searches everything, so unit_00 chunks are always included naturally.
"""

import json
from utils.syllabus_lookup import (
    load_syllabus,
    get_all_units,
    get_topics,
    get_unit_title,
)
from utils.llm_client import generate_json
from ingestion.topic_resolver import UNMATCHED_UNIT_KEY


def _build_syllabus_summary(syllabus: dict) -> str:
    lines = []
    for course_key, course_data in syllabus.items():
        lines.append(f"Course: {course_data['title']} (key: {course_key})")
        for unit_key, unit_data in course_data["units"].items():
            topics_str = ", ".join(unit_data["topics"])
            lines.append(
                f"  {unit_key}: {unit_data['title']}"
                f"\n    Topics: {topics_str}"
            )
    return "\n".join(lines)


def get_all_course_topics(syllabus: dict, course_key: str) -> dict:
    course = syllabus.get(course_key, {})
    result = {}
    for unit_key, unit_data in course.get("units", {}).items():
        result[unit_key] = {
            "title": unit_data["title"],
            "topics": unit_data["topics"]
        }
    return result


def detect_course_and_unit(
    query: str,
    syllabus: dict = None
) -> dict:
    """
    Classifies a student query into course, unit, and topic.
    Uses local Ollama LLM — no Gemini quota consumed.
    """
    if syllabus is None:
        syllabus = load_syllabus()

    syllabus_summary = _build_syllabus_summary(syllabus)

    prompt = (
        "You are a course assistant. A student has submitted a query. "
        "Using the syllabus below, identify:\n"
        "1. Which course the query relates to (course_key)\n"
        "2. Which unit within that course (unit_key)\n"
        "3. Which specific topic within that unit (topic)\n\n"
        f"Syllabus:\n{syllabus_summary}\n\n"
        f"Student query: {query}\n\n"
        "Return JSON with keys: course_key, unit_key, topic, confidence.\n"
        "Use null for any field you cannot determine.\n"
        'Example: {"course_key": "deep_learning", "unit_key": "unit_2", '
        '"topic": "max pooling layer", "confidence": "high"}'
    )

    try:
        raw = generate_json(prompt)
        parsed = json.loads(raw)

        course_key = parsed.get("course_key")
        unit_key = parsed.get("unit_key")
        topic = parsed.get("topic")
        confidence = parsed.get("confidence", "low")

        if course_key not in syllabus:
            course_key = None

        if course_key and unit_key:
            valid_units = [u["key"] for u in get_all_units(syllabus, course_key)]
            if unit_key not in valid_units:
                unit_key = None
                topic = None

        if course_key and unit_key and topic:
            filter_scope = "topic"
        elif course_key and unit_key:
            filter_scope = "unit"
        elif course_key:
            filter_scope = "course"
        else:
            filter_scope = "none"

        course_data = syllabus.get(course_key, {}) if course_key else {}
        unit_topics = (
            get_topics(syllabus, course_key, unit_key)
            if course_key and unit_key else []
        )
        all_course_topics = (
            get_all_course_topics(syllabus, course_key)
            if course_key else {}
        )

        return {
            "course_key": course_key,
            "course_title": course_data.get("title", ""),
            "unit_key": unit_key,
            "unit_title": (
                get_unit_title(syllabus, course_key, unit_key)
                if course_key and unit_key else ""
            ),
            "topic": topic,
            "topics": unit_topics,
            "all_course_topics": all_course_topics,
            "confidence": confidence,
            "filter_scope": filter_scope
        }

    except Exception as e:
        print(f"Warning: detection failed. Reason: {e}")
        return {
            "course_key": None,
            "course_title": "",
            "unit_key": None,
            "unit_title": "",
            "topic": None,
            "topics": [],
            "all_course_topics": {},
            "confidence": "low",
            "filter_scope": "none"
        }


def build_chroma_filter(detection: dict, mode: str = "qa") -> dict | None:
    """
    Converts a detection result into a ChromaDB metadata filter.

    Filter logic by scenario:

    Unit detected (scope = topic or unit):
        Search the detected unit AND unit_00 so unclassified chunks
        are always included. This is the key behaviour change.

    Course only detected (scope = course):
        Search the whole course including unit_00.

    Practice test mode (quiz):
        Always search the whole course so all units are covered.

    Nothing detected (scope = none):
        No filter, search everything across all documents.
    """
    scope = detection.get("filter_scope", "none")
    course_key = detection.get("course_key")
    unit_key = detection.get("unit_key")

    if not course_key:
        return None

    # Practice test always covers the whole course
    if mode == "quiz":
        return {"course_key": {"$eq": course_key}}

    # Unit detected: search that unit plus the catch-all unit_00
    if scope in ("topic", "unit") and unit_key:
        return {
            "$and": [
                {"course_key": {"$eq": course_key}},
                {
                    "unit_key": {
                        "$in": [unit_key, UNMATCHED_UNIT_KEY]
                    }
                }
            ]
        }

    # Course only: search entire course including unit_00
    return {"course_key": {"$eq": course_key}}
