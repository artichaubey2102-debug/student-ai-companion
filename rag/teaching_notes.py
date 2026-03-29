"""
teaching_notes.py
-----------------
Generates structured teaching notes for professors using local Ollama.
"""

import os
from dotenv import load_dotenv
from utils.llm_client import generate
from rag.retriever import retrieve_cosine
from rag.prompts import teaching_notes_prompt
from utils.syllabus_lookup import load_syllabus, get_topics, get_unit_title

load_dotenv()


def generate_teaching_notes(
    course_key: str,
    unit_key: str,
    syllabus: dict = None,
    top_k: int = 8
) -> dict:
    """
    Generates structured teaching notes for a given course unit.
    Uses local Ollama for generation, Gemini only for the retrieval embedding.

    Args:
        course_key: e.g. 'deep_learning'
        unit_key:   e.g. 'unit_2'
        syllabus:   Pre-loaded syllabus dict. Loaded if None.
        top_k:      Number of chunks to retrieve for context.

    Returns:
        Dict with: notes, course_title, unit_title, topics, chunks_used.
    """
    if syllabus is None:
        syllabus = load_syllabus()

    course_title = syllabus[course_key]["title"]
    unit_title = get_unit_title(syllabus, course_key, unit_key)
    topics = get_topics(syllabus, course_key, unit_key)

    retrieval_query = f"{unit_title}: {', '.join(topics[:6])}"

    chroma_filter = {
        "$and": [
            {"course_key": {"$eq": course_key}},
            {"unit_key": {"$eq": unit_key}}
        ]
    }

    chunks = retrieve_cosine(
        query=retrieval_query,
        chroma_filter=chroma_filter,
        top_k=top_k,
        strategy="overlap"
    )

    prompt = teaching_notes_prompt(course_title, unit_title, topics, chunks)
    notes_text = generate(prompt)

    if not notes_text:
        notes_text = (
            "Unable to generate teaching notes. "
            "Ensure Ollama is running and llama3.2 is installed."
        )

    return {
        "notes": notes_text,
        "course_title": course_title,
        "unit_title": unit_title,
        "topics": topics,
        "chunks_used": len(chunks)
    }
