"""
topic_resolver.py
-----------------
Maps text chunks to syllabus units using keyword matching.

Two text sources are scored for each chunk:
    1. The chunk body text (standard keyword matching)
    2. The page heading extracted by pdf_extractor (chapter/section title)

Headings are given 3x weight because a chapter titled "Convolutional
Neural Networks" is a much stronger signal than a body paragraph that
happens to mention "convolution" once. This approach significantly
reduces the number of chunks assigned to unit_00.

Matching order:
    1. If heading score is high enough, use heading result directly.
    2. Otherwise combine heading score (weighted) + body score.
    3. If combined score still zero, assign to unit_00.
"""

from utils.syllabus_lookup import (
    get_all_units,
    get_unit_title,
    get_keywords,
    get_topics,
)

UNMATCHED_UNIT_KEY = "unit_00"
UNMATCHED_UNIT_TITLE = "General and Unclassified Content"

_HEADING_WEIGHT = 3
_KEYWORD_THRESHOLD = 1


def _score_text(text: str, keywords: list[str]) -> int:
    """Counts keyword matches in a text string."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _score_against_topics(text: str, topics: list[str]) -> int:
    """
    Scores text against unit topic names directly.
    Topic names are more descriptive than single keywords and match
    chapter headings well. For example a heading 'Max Pooling Layer'
    will match the topic 'max pooling layer' exactly.
    """
    text_lower = text.lower()
    score = 0
    for topic in topics:
        topic_words = topic.lower().split()
        if len(topic_words) >= 2:
            if topic.lower() in text_lower:
                score += 2
        else:
            if topic.lower() in text_lower:
                score += 1
    return score


def resolve_unit(
    chunk_text: str,
    syllabus: dict,
    course_key: str,
    heading: str = ""
) -> dict:
    """
    Resolves a chunk to the best matching syllabus unit.

    Uses both the chunk body text and the page heading as signals.
    The heading receives 3x weight because chapter and section titles
    are far more precise indicators of content than body text.

    Args:
        chunk_text:  Body text of the chunk.
        syllabus:    Loaded syllabus dictionary.
        course_key:  Course this chunk belongs to.
        heading:     Chapter or section heading from pdf_extractor.
                     Empty string if no heading was detected.

    Returns:
        Dict with unit_key, unit_title, resolution_method.
    """
    units = get_all_units(syllabus, course_key)
    best_unit = None
    best_score = 0

    heading_clean = heading.strip().lower() if heading else ""

    for unit in units:
        unit_key = unit["key"]
        keywords = get_keywords(syllabus, course_key, unit_key)
        topics = get_topics(syllabus, course_key, unit_key)

        # Score body text against keywords
        body_score = _score_text(chunk_text, keywords)

        # Score heading against keywords and topic names
        heading_kw_score = _score_text(heading_clean, keywords)
        heading_topic_score = _score_against_topics(heading_clean, topics)
        heading_score = (heading_kw_score + heading_topic_score) * _HEADING_WEIGHT

        combined = body_score + heading_score

        if combined > best_score:
            best_score = combined
            best_unit = unit_key

    if best_score >= _KEYWORD_THRESHOLD and best_unit:
        method = "heading+keyword" if heading_clean else "keyword"
        return {
            "unit_key": best_unit,
            "unit_title": get_unit_title(syllabus, course_key, best_unit),
            "resolution_method": method
        }
    else:
        return {
            "unit_key": UNMATCHED_UNIT_KEY,
            "unit_title": UNMATCHED_UNIT_TITLE,
            "resolution_method": "unmatched"
        }
