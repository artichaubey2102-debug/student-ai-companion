"""
rag_engine.py
-------------
Orchestrates the full student query pipeline.
Source references are appended after LLM generation using chunk metadata.
"""

import os
from dotenv import load_dotenv
from utils.llm_client import generate
from rag.query_detector import detect_course_and_unit, build_chroma_filter
from rag.retriever import retrieve
from rag.prompts import build_prompt, get_sources

load_dotenv()


def answer(
    query: str,
    mode: str,
    retrieval_method: str = "cosine",
    chunking_strategy: str = "overlap",
    top_k: int = 5,
    syllabus: dict = None
) -> dict:
    """
    Runs the full RAG pipeline for a student query.

    Returns:
        Dict with: response, sources, detection, chunks, mode, query.
        'response' is the clean LLM answer without internal references.
        'sources' is a formatted string of book and chapter references.
    """
    detection = detect_course_and_unit(query, syllabus)
    chroma_filter = build_chroma_filter(detection, mode=mode)
    effective_top_k = 10 if mode == "quiz" else top_k

    chunks = retrieve(
        query=query,
        chroma_filter=chroma_filter,
        method=retrieval_method,
        top_k=effective_top_k,
        strategy=chunking_strategy
    )

    prompt = build_prompt(mode, query, chunks, detection)
    response_text = generate(prompt)

    if not response_text:
        response_text = (
            "The system was unable to generate a response. "
            "Ensure Ollama is running and llama3.2 is installed."
        )

    sources = get_sources(chunks)

    return {
        "response": response_text,
        "sources": sources,
        "detection": detection,
        "chunks": chunks,
        "mode": mode,
        "query": query
    }
