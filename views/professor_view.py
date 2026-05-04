"""
professor_view.py
-----------------
Professor portal. Teaching Notes first, Knowledge Base Status second.
"""

import streamlit as st
import os
from utils.syllabus_lookup import get_all_courses, get_all_units, get_topics
from rag.teaching_notes import generate_teaching_notes
import markdown as md_lib


def _render_answer(text: str, label: str = "Notes"):
    try:
        import markdown
        html_body = markdown.markdown(
            text, extensions=["fenced_code", "tables", "nl2br"]
        )
    except Exception:
        escaped = text.replace("&", "&amp;").replace("<", "&lt;")
        html_body = "".join(
            f"<p>{line}</p>" for line in escaped.split("\n") if line.strip()
        )
    st.markdown(
        f'<div class="answer-block">'
        f'<div class="answer-block-label">{label}</div>'
        f'{html_body}</div>',
        unsafe_allow_html=True
    )


def _check_password() -> bool:
    prof_password = os.environ.get("PROFESSOR_PASSWORD", "")
    if not prof_password:
        st.error("PROFESSOR_PASSWORD not set in .env or Streamlit secrets.")
        return False
    if st.session_state.get("professor_authenticated"):
        return True

    st.markdown(
        "<p style='font-size:0.95rem;color:#4B5563;margin:0 0 1rem 0'>"
        "Enter your professor password to access teaching tools.</p>",
        unsafe_allow_html=True
    )
    password = st.text_input("Password", type="password", key="prof_pw")
    if st.button("Sign in", key="prof_signin", type="primary"):
        if password == prof_password:
            st.session_state.professor_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _teaching_notes(syllabus: dict):
    st.markdown(
        "<p style='color:#4B5563;line-height:1.65;margin:0 0 1.2rem 0'>"
        "Select a course and unit to generate structured teaching notes "
        "covering key concepts, common student misconceptions, suggested "
        "discussion questions, and content gap analysis.</p>",
        unsafe_allow_html=True
    )

    course_options = {c["title"]: c["key"] for c in get_all_courses(syllabus)}
    col1, col2 = st.columns(2)
    with col1:
        course_title = st.selectbox(
            "Course", list(course_options.keys()), key="notes_course"
        )
    course_key = course_options[course_title]
    unit_options = {
        u["title"]: u["key"] for u in get_all_units(syllabus, course_key)
    }
    with col2:
        unit_title_sel = st.selectbox(
            "Unit", list(unit_options.keys()), key="notes_unit"
        )
    unit_key = unit_options[unit_title_sel]
    topics = get_topics(syllabus, course_key, unit_key)
    if topics:
        st.markdown(
            f"<p style='font-size:0.78rem;color:#9CA3AF;margin:4px 0 1rem 0'>"
            f"Topics: {', '.join(topics[:7])}"
            f"{'&nbsp;...' if len(topics) > 7 else ''}</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("")

    if st.button("Generate Teaching Notes", key="notes_btn", type="primary"):
        with st.spinner(f"Generating notes for {unit_title_sel}..."):
            result = generate_teaching_notes(
                course_key=course_key,
                unit_key=unit_key,
                syllabus=syllabus
            )
            st.session_state.notes_result = result

    if st.session_state.get("notes_result"):
        r = st.session_state.notes_result
        st.markdown("")
        st.markdown(
            f"<p style='font-size:0.72rem;color:#9CA3AF;margin:0 0 0.3rem 0'>"
            f"Generated from {r['chunks_used']} content excerpts</p>",
            unsafe_allow_html=True
        )
        _render_answer(r["notes"], "Teaching Notes")
        st.markdown("")
        st.download_button(
            label="Download as text file",
            data=r["notes"],
            file_name=f"teaching_notes_{course_key}_{unit_key}.txt",
            mime="text/plain",
            key="notes_download"
        )


def _kb_status():
    st.markdown(
        "<p style='color:#4B5563;line-height:1.65;margin:0 0 1.2rem 0'>"
        "Chunk counts across all three ChromaDB collections.</p>",
        unsafe_allow_html=True
    )
    try:
        import chromadb
        import pandas as pd
        from collections import Counter

        chroma_path = os.path.join(
            os.path.dirname(__file__), "..", "chroma_db"
        )
        client = chromadb.PersistentClient(path=chroma_path)
        rows = []
        for strategy in ["fixed", "overlap", "semantic"]:
            try:
                col = client.get_collection(f"chunks_{strategy}")
                rows.append({
                    "Collection": f"chunks_{strategy}",
                    "Chunks": col.count(),
                    "Status": "Ready"
                })
            except Exception:
                rows.append({
                    "Collection": f"chunks_{strategy}",
                    "Chunks": 0,
                    "Status": "Empty"
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        try:
            col = client.get_collection("chunks_overlap")
            results = col.get(include=["metadatas"])
            if results["metadatas"]:
                courses = Counter(
                    m["course_title"] for m in results["metadatas"]
                )
                st.markdown("")
                st.caption("By course (overlap collection):")
                st.dataframe(
                    pd.DataFrame([
                        {"Course": k, "Chunks": v}
                        for k, v in sorted(courses.items())
                    ]),
                    use_container_width=True, hide_index=True
                )
        except Exception:
            pass
    except Exception:
        st.info("ChromaDB unavailable. Run ingestion first.")


def render_professor_portal(syllabus: dict):
    st.markdown(
        "<h2 style='font-size:1.5rem;font-weight:700;margin:0 0 1.2rem 0'>"
        "Professor Portal</h2>",
        unsafe_allow_html=True
    )
    if not _check_password():
        return

    tab_notes, tab_status = st.tabs([
        "Teaching Notes", "Knowledge Base Status"
    ])
    with tab_notes:
        st.markdown("")
        _teaching_notes(syllabus)
    with tab_status:
        st.markdown("")
        _kb_status()
