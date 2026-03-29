"""
prompts.py
----------
Prompt templates for the four student modes and professor teaching notes.

Context chunks are passed to the LLM as clean text without excerpt labels
so the model never references internal retrieval identifiers in its response.
Source references are appended separately after generation using chunk metadata.
"""


def _format_context_for_llm(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks as clean text for the LLM prompt.
    No excerpt numbers or source labels are included so the model
    does not reference them in its response.
    """
    if not chunks:
        return "No relevant content was found in the knowledge base."
    return "\n\n---\n\n".join(chunk["text"] for chunk in chunks)


def _format_sources(chunks: list[dict]) -> str:
    """
    Builds a clean source reference block from chunk metadata.
    Uses book filename, chapter heading, and unit title.
    This is appended to the response after LLM generation.
    """
    if not chunks:
        return ""

    seen = set()
    sources = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        source_file = meta.get("source_file", "")
        heading = meta.get("heading", "")
        unit_title = meta.get("unit_title", "")

        # Clean up filename for display
        display_name = source_file.replace("_", " ").replace(".pdf", "").strip()

        key = (display_name, heading)
        if key in seen:
            continue
        seen.add(key)

        if heading and heading != "General and Unclassified Content":
            sources.append(f"{display_name} — {heading}")
        elif unit_title and unit_title != "General and Unclassified Content":
            sources.append(f"{display_name} — {unit_title}")
        else:
            sources.append(display_name)

    if not sources:
        return ""

    lines = ["Sources:"]
    for i, s in enumerate(sources, 1):
        lines.append(f"  {i}. {s}")

    return "\n".join(lines)


def _scope_line(detection: dict) -> str:
    topic = detection.get("topic")
    unit_title = detection.get("unit_title")
    course_title = detection.get("course_title")

    if topic and unit_title and course_title:
        return (
            f"The student is asking about '{topic}' "
            f"from {course_title}, {unit_title}.\n"
        )
    elif unit_title and course_title:
        topics = detection.get("topics", [])
        topics_str = ", ".join(topics[:6]) if topics else ""
        return (
            f"The student is asking about {course_title}, {unit_title}.\n"
            + (f"Topics in this unit: {topics_str}.\n" if topics_str else "")
        )
    elif course_title:
        return f"The student is asking about {course_title}.\n"
    return ""


def _all_course_topics_block(detection: dict) -> str:
    all_topics = detection.get("all_course_topics", {})
    if not all_topics:
        return ""
    lines = ["Full course syllabus topics:"]
    for unit_key in sorted(all_topics.keys()):
        unit_data = all_topics[unit_key]
        unit_title = unit_data.get("title", unit_key)
        topics = unit_data.get("topics", [])
        lines.append(f"  {unit_title}: {', '.join(topics)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode 1: Question and Answer
# ---------------------------------------------------------------------------

def qa_prompt(query: str, chunks: list[dict], detection: dict) -> str:
    context = _format_context_for_llm(chunks)
    scope = _scope_line(detection)

    return (
        "You are a knowledgeable academic assistant helping a postgraduate "
        "Data Science student.\n"
        f"{scope}"
        "Answer the student's question using only the provided course material "
        "below. Be thorough but concise. Do not reference any excerpt numbers, "
        "source labels, or internal document identifiers in your answer. "
        "Write as if explaining from your own knowledge.\n"
        "If the answer is not fully covered in the material, say so clearly.\n\n"
        f"Course material:\n{context}\n\n"
        f"Student question: {query}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Mode 2: Flashcards
# ---------------------------------------------------------------------------

def flashcard_prompt(query: str, chunks: list[dict], detection: dict) -> str:
    context = _format_context_for_llm(chunks)
    scope = _scope_line(detection)

    return (
        "You are an academic tutor creating study flashcards for a postgraduate "
        "Data Science student.\n"
        f"{scope}"
        "Using only the course material below, generate 6 flashcards covering "
        "the most important concepts related to the student's topic.\n\n"
        "Format each flashcard exactly as:\n"
        "CARD 1\n"
        "Front: [concept or term]\n"
        "Back: [clear explanation in 2-3 sentences]\n\n"
        "Do not reference excerpt numbers or source labels anywhere. "
        "Only use information present in the course material.\n\n"
        f"Course material:\n{context}\n\n"
        f"Topic: {query}\n\n"
        "Flashcards:"
    )


# ---------------------------------------------------------------------------
# Mode 3: Practice Test
# ---------------------------------------------------------------------------

def practice_test_prompt(
    query: str,
    chunks: list[dict],
    detection: dict
) -> str:
    context = _format_context_for_llm(chunks)
    scope = _scope_line(detection)
    course_topics_block = _all_course_topics_block(detection)
    course_title = detection.get("course_title", "the course")

    return (
        "You are an academic examiner setting a comprehensive practice test "
        "for a postgraduate Data Science student.\n"
        f"{scope}"
        f"Generate a full practice test for {course_title} covering topics "
        "from all units of the course.\n\n"
        f"{course_topics_block}\n\n"
        "Use the course material below as the primary content source. "
        "Do not reference excerpt numbers or source labels anywhere.\n\n"
        "Structure the test as follows:\n\n"
        "Section A: Multiple Choice (1 mark each)\n"
        "6 questions covering different units, 4 options each (A B C D), "
        "correct answer marked at the end of each question.\n\n"
        "Section B: Short Answer (5 marks each)\n"
        "3 questions requiring 3-5 sentence responses, one from each of "
        "three different units.\n\n"
        "Section C: Long Answer (10 marks each)\n"
        "1 analytical question integrating concepts from at least two units.\n\n"
        f"Course material:\n{context}\n\n"
        "Practice Test:"
    )


# ---------------------------------------------------------------------------
# Mode 4: Summary
# ---------------------------------------------------------------------------

def summary_prompt(query: str, chunks: list[dict], detection: dict) -> str:
    context = _format_context_for_llm(chunks)
    scope = _scope_line(detection)

    return (
        "You are an academic tutor writing a concise study summary for a "
        "postgraduate Data Science student.\n"
        f"{scope}"
        "Using only the course material below, write a structured summary "
        "with these sections:\n"
        "- One paragraph overview\n"
        "- Key concepts with brief explanations\n"
        "- Important relationships between concepts\n"
        "- 4 to 5 key takeaways\n\n"
        "Do not reference excerpt numbers or source labels. "
        "Write in clear academic prose.\n\n"
        f"Course material:\n{context}\n\n"
        f"Topic: {query}\n\n"
        "Summary:"
    )


# ---------------------------------------------------------------------------
# Professor mode: Teaching notes
# ---------------------------------------------------------------------------

def teaching_notes_prompt(
    course_title: str,
    unit_title: str,
    topics: list[str],
    chunks: list[dict]
) -> str:
    context = _format_context_for_llm(chunks)
    topics_str = ", ".join(topics) if topics else "all topics in this unit"

    return (
        "You are an experienced university lecturer preparing teaching notes.\n"
        f"Course: {course_title}\n"
        f"Unit: {unit_title}\n"
        f"Topics: {topics_str}\n\n"
        "Using the course material below, produce structured teaching notes "
        "with these sections:\n\n"
        "1. Unit overview (2-3 sentences)\n"
        "2. Key concepts to emphasise (5-7 ideas with a teaching note each)\n"
        "3. Common student misconceptions (3-4 with suggested clarifications)\n"
        "4. Suggested discussion questions (3 seminar questions)\n"
        "5. Content gaps (topics in the syllabus with limited material coverage)\n\n"
        "Do not reference excerpt numbers or source labels.\n\n"
        f"Course material:\n{context}\n\n"
        "Teaching Notes:"
    )


# ---------------------------------------------------------------------------
# Source reference appended after LLM response
# ---------------------------------------------------------------------------

def get_sources(chunks: list[dict]) -> str:
    """
    Returns a formatted source reference string built from chunk metadata.
    Called by rag_engine.py after generation and appended to the response.
    """
    return _format_sources(chunks)


# ---------------------------------------------------------------------------
# Prompt router
# ---------------------------------------------------------------------------

def build_prompt(
    mode: str,
    query: str,
    chunks: list[dict],
    detection: dict
) -> str:
    if mode == "qa":
        return qa_prompt(query, chunks, detection)
    elif mode == "flashcard":
        return flashcard_prompt(query, chunks, detection)
    elif mode == "quiz":
        return practice_test_prompt(query, chunks, detection)
    elif mode == "summary":
        return summary_prompt(query, chunks, detection)
    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. "
            "Choose from 'qa', 'flashcard', 'quiz', or 'summary'."
        )
