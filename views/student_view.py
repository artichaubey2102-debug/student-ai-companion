"""
student_view.py
---------------
Student portal. Uses st.radio styled as pill tabs for mode selection.
Answer content is rendered as a single HTML block to avoid broken div issues.
"""

import streamlit as st
import markdown as md_lib
from utils.syllabus_lookup import get_all_courses, get_all_units
from rag.rag_engine import answer


def _init_state():
    # Only initialise result keys and active mode.
    # Text input keys (qa_query etc.) are intentionally NOT initialised here
    # so that Streamlit's widget state management preserves typed text
    # across reruns triggered by Generate button clicks.
    defaults = {
        "active_mode": "qa",
        "qa_result": None,
        "fl_result": None,
        "quiz_result": None,
        "sm_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _course_opts(syllabus):
    return {c["title"]: c["key"] for c in get_all_courses(syllabus)}


def _unit_opts(syllabus, course_key):
    d = {"All units": None}
    d.update({u["title"]: u["key"] for u in get_all_units(syllabus, course_key)})
    return d


def _render_mode_selector():
    """
    Renders four mode tiles as st.buttons.
    Active tile = primary (blue), inactive = secondary (outlined).
    Title only in button — no description text to avoid concatenation.
    Description shown as caption below tiles.
    """
    st.markdown("""
    <style>
    /* Tile buttons */
    div.tile-row div[data-testid="stButton"] button {
        height: 52px !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        transition: all 0.15s !important;
    }
    div.tile-row div[data-testid="stButton"] button[kind="secondary"] {
        border: 1.5px solid #D1D5DB !important;
        color: #374151 !important;
        background: white !important;
    }
    div.tile-row div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background: #EFF6FF !important;
    }
    </style>
    <div class="tile-row">
    """, unsafe_allow_html=True)

    modes = [
        ("qa", "Question & Answer"),
        ("flashcard", "Flashcards"),
        ("quiz", "Practice Test"),
        ("summary", "Summary"),
    ]
    descs = {
        "qa": "Ask any question from your course material",
        "flashcard": "Generate study flashcards for any topic",
        "quiz": "Full test covering all units of a course",
        "summary": "Structured summary with key concepts",
    }

    active = st.session_state.active_mode
    cols = st.columns(4)
    for i, (key, label) in enumerate(modes):
        with cols[i]:
            btn_type = "primary" if key == active else "secondary"
            if st.button(label, key=f"mode_{key}", type=btn_type,
                         use_container_width=True):
                st.session_state.active_mode = key
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Show description for active mode
    active_desc = descs.get(active, "")
    st.markdown(
        f"<p style='font-size:0.8rem;color:#6B7280;margin:6px 0 1.2rem 2px'>"
        f"{active_desc}</p>",
        unsafe_allow_html=True
    )


def _scope_pill(detection):
    course = detection.get("course_title", "")
    unit = detection.get("unit_title", "")
    topic = detection.get("topic", "")
    scope = detection.get("filter_scope", "none")
    if scope == "none" or not course:
        label = "All course material"
    else:
        parts = [course]
        if unit:
            parts.append(unit)
        if topic:
            parts.append(topic)
        label = " › ".join(parts)
    st.markdown(
        f'<span class="scope-pill">&#x1F4CD; {label}</span>',
        unsafe_allow_html=True
    )


def _render_answer(text: str, label: str = "Answer"):
    """
    Renders the answer as a single self-contained HTML block.
    Converts markdown to HTML internally so the entire block is
    one st.markdown call — no open/close div mismatch.
    """
    try:
        import markdown
        html_body = markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "nl2br"]
        )
    except Exception:
        # Fallback: escape and wrap in pre-style paragraphs
        escaped = text.replace("&", "&amp;").replace("<", "&lt;")
        html_body = "".join(
            f"<p>{line}</p>" for line in escaped.split("\n") if line.strip()
        )

    st.markdown(
        f'<div class="answer-block">'
        f'<div class="answer-block-label">{label}</div>'
        f'{html_body}'
        f'</div>',
        unsafe_allow_html=True
    )


def _sources(result):
    if result.get("sources"):
        with st.expander("View sources", expanded=False):
            st.caption(result["sources"])


def _course_unit_row(syllabus, pfx, course_opts):
    c1, c2 = st.columns(2)
    with c1:
        course = st.selectbox(
            "Course (optional)",
            ["Auto-detect"] + list(course_opts.keys()),
            key=f"{pfx}_course"
        )
    with c2:
        if course != "Auto-detect":
            st.selectbox(
                "Unit (optional)",
                list(_unit_opts(syllabus, course_opts[course]).keys()),
                key=f"{pfx}_unit"
            )
        else:
            st.selectbox(
                "Unit (optional)", ["Select a course first"],
                disabled=True, key=f"{pfx}_unit_dis"
            )
    return course


def _mode_qa(syllabus, course_opts):
    course = _course_unit_row(syllabus, "qa", course_opts)
    st.text_area(
        "Your question",
        placeholder="e.g. How does the vanishing gradient problem affect RNNs?",
        height=110, key="qa_query"
    )
    if st.button("Get Answer", key="qa_generate", type="primary"):
        q = st.session_state.qa_query.strip()
        if not q:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching course material..."):
                pfx = f"{course}: " if course != "Auto-detect" else ""
                st.session_state.qa_result = answer(pfx + q, mode="qa")

    if st.session_state.qa_result:
        st.markdown("")
        _scope_pill(st.session_state.qa_result["detection"])
        _render_answer(st.session_state.qa_result["response"], "Answer")
        _sources(st.session_state.qa_result)


def _mode_flashcards(syllabus, course_opts):
    course = _course_unit_row(syllabus, "fl", course_opts)
    st.text_area(
        "Topic for flashcards",
        placeholder="e.g. LSTM gates and memory cell",
        height=110, key="fl_query"
    )
    if st.button("Generate Flashcards", key="fl_generate", type="primary"):
        q = st.session_state.fl_query.strip()
        if not q:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating flashcards..."):
                pfx = f"{course}: " if course != "Auto-detect" else ""
                st.session_state.fl_result = answer(
                    pfx + q, mode="flashcard"
                )

    if st.session_state.fl_result:
        st.markdown("")
        _scope_pill(st.session_state.fl_result["detection"])
        raw = st.session_state.fl_result["response"]
        cards, cur = [], {}
        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("CARD"):
                if cur.get("front") and cur.get("back"):
                    cards.append(cur)
                cur = {}
            elif line.startswith("Front:"):
                cur["front"] = line.replace("Front:", "").strip()
            elif line.startswith("Back:"):
                cur["back"] = line.replace("Back:", "").strip()
        if cur.get("front") and cur.get("back"):
            cards.append(cur)

        if not cards:
            _render_answer(raw, "Flashcards")
        else:
            st.markdown(
                f"<p style='font-size:0.85rem;color:#6B7280;margin:0 0 0.6rem 0'>"
                f"{len(cards)} flashcards generated. "
                f"Click a card to reveal the answer.</p>",
                unsafe_allow_html=True
            )
            for i, card in enumerate(cards, 1):
                with st.expander(
                    f"Card {i}  —  {card['front']}", expanded=False
                ):
                    st.markdown(card["back"])
        _sources(st.session_state.fl_result)


def _mode_quiz(syllabus, course_opts):
    quiz_course = st.selectbox(
        "Select course", list(course_opts.keys()), key="quiz_course"
    )
    st.text_area(
        "Additional instructions (optional)",
        placeholder="e.g. Focus more on practical applications. "
                    "Leave blank for a balanced test.",
        height=80, key="quiz_extra"
    )
    if st.button("Generate Practice Test", key="quiz_generate", type="primary"):
        with st.spinner("Generating practice test..."):
            extra = st.session_state.quiz_extra.strip()
            q = f"Generate a practice test for {quiz_course}."
            if extra:
                q += f" {extra}"
            st.session_state.quiz_result = answer(q, mode="quiz")

    if st.session_state.quiz_result:
        st.markdown("")
        _scope_pill(st.session_state.quiz_result["detection"])
        _render_answer(
            st.session_state.quiz_result["response"], "Practice Test"
        )
        _sources(st.session_state.quiz_result)


def _mode_summary(syllabus, course_opts):
    course = _course_unit_row(syllabus, "sm", course_opts)
    st.text_area(
        "Topic to summarise",
        placeholder="e.g. RDD operations in Apache Spark",
        height=110, key="sm_query"
    )
    if st.button("Generate Summary", key="sm_generate", type="primary"):
        q = st.session_state.sm_query.strip()
        if not q:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating summary..."):
                pfx = f"{course}: " if course != "Auto-detect" else ""
                st.session_state.sm_result = answer(
                    pfx + q, mode="summary"
                )

    if st.session_state.sm_result:
        st.markdown("")
        _scope_pill(st.session_state.sm_result["detection"])
        _render_answer(st.session_state.sm_result["response"], "Summary")
        _sources(st.session_state.sm_result)


def render_student_portal(syllabus: dict):
    _init_state()

    st.markdown(
        "<h2 style='font-size:1.5rem;font-weight:700;margin:0 0 0.4rem 0'>"
        "Student Portal</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#4B5563;line-height:1.65;margin:0 0 0.4rem 0'>"
        "A study companion built for M.Sc. Data Science and Analytics "
        "students at Devi Ahilya Vishwavidyalaya, Indore.</p>"
        "<p style='color:#4B5563;line-height:1.65;margin:0 0 1.4rem 0'>"
        "Unlike general-purpose tools such as ChatGPT or Gemini, this "
        "system grounds every answer, flashcard, practice test, and summary "
        "in the actual syllabus and reference texts used in your programme. "
        "You get more precise, course-specific study support than any "
        "general-purpose AI can provide.</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;color:#374151;"
        "text-transform:uppercase;letter-spacing:0.06em;margin:0 0 0.6rem 0'>"
        "Study Mode</p>",
        unsafe_allow_html=True
    )

    _render_mode_selector()

    course_opts = _course_opts(syllabus)
    mode = st.session_state.active_mode

    st.markdown(
        "<hr style='border:none;border-top:1px solid #F3F4F6;"
        "margin:0 0 1.2rem 0'>",
        unsafe_allow_html=True
    )

    if mode == "qa":
        _mode_qa(syllabus, course_opts)
    elif mode == "flashcard":
        _mode_flashcards(syllabus, course_opts)
    elif mode == "quiz":
        _mode_quiz(syllabus, course_opts)
    elif mode == "summary":
        _mode_summary(syllabus, course_opts)
