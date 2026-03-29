import json
import os

_SYLLABUS_PATH = os.path.join(os.path.dirname(__file__), "..", "syllabus.json")

def load_syllabus() -> dict:
    with open(_SYLLABUS_PATH, "r") as f:
        return json.load(f)

def get_all_courses(syllabus: dict) -> list:
    return [
        {"key": k, "title": v["title"], "code": v["code"]}
        for k, v in syllabus.items()
    ]

def get_all_units(syllabus: dict, course_key: str) -> list:
    course = syllabus.get(course_key, {})
    return [
        {"key": k, "title": v["title"]}
        for k, v in course.get("units", {}).items()
    ]

def get_topics(syllabus: dict, course_key: str, unit_key: str) -> list:
    return (
        syllabus
        .get(course_key, {})
        .get("units", {})
        .get(unit_key, {})
        .get("topics", [])
    )

def get_keywords(syllabus: dict, course_key: str, unit_key: str) -> list:
    return (
        syllabus
        .get(course_key, {})
        .get("units", {})
        .get(unit_key, {})
        .get("keywords", [])
    )

def get_unit_title(syllabus: dict, course_key: str, unit_key: str) -> str:
    return (
        syllabus
        .get(course_key, {})
        .get("units", {})
        .get(unit_key, {})
        .get("title", "")
    )

def get_all_keywords_flat(syllabus: dict, course_key: str) -> dict:
    """
    Returns a mapping of keyword -> unit_key for a given course.
    Used by the topic resolver to match a chunk to a unit quickly.
    """
    mapping = {}
    course = syllabus.get(course_key, {})
    for unit_key, unit_data in course.get("units", {}).items():
        for kw in unit_data.get("keywords", []):
            mapping[kw.lower()] = unit_key
    return mapping
