import streamlit as st
from dotenv import load_dotenv
import os
import json

load_dotenv()

st.set_page_config(
    page_title="Student AI Companion",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_syllabus():
    path = os.path.join(os.path.dirname(__file__), "syllabus.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as f:
            return json.load(f)
    return {}

syllabus = load_syllabus()

from views.student_view import render_student_portal
from views.professor_view import render_professor_portal

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* Hide Streamlit top toolbar (Deploy button bar) */
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
#MainMenu { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 860px !important;
}

/* Global button reset - use Streamlit's blue, not red */
button[kind="primary"] {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
}
button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

/* Scope pill */
.scope-pill {
    display: inline-block;
    font-size: 0.72rem;
    padding: 3px 11px;
    border-radius: 20px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1D4ED8;
    margin-bottom: 1rem;
    font-weight: 500;
}

/* Answer block - rendered as single HTML unit */
.answer-block {
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    background: #F8FBFF;
    padding: 1.4rem 1.6rem 1.5rem 1.6rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.answer-block-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #93C5FD;
    margin-bottom: 1rem;
}
.answer-block p { margin-bottom: 0.75rem; line-height: 1.75; }
.answer-block h1,.answer-block h2,.answer-block h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 1rem 0 0.4rem 0;
}
.answer-block ul,.answer-block ol {
    margin: 0.4rem 0 0.75rem 1.2rem;
    line-height: 1.75;
}
.answer-block strong { font-weight: 600; }
.answer-block code {
    background: #E0F2FE;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.88em;
}

@media (prefers-color-scheme: dark) {
    .scope-pill {
        background: rgba(37,99,235,0.15);
        border-color: rgba(96,165,250,0.3);
        color: #93C5FD;
    }
    .answer-block {
        border-color: rgba(96,165,250,0.2);
        background: rgba(37,99,235,0.04);
    }
    .answer-block-label { color: rgba(96,165,250,0.5); }
    .answer-block code { background: rgba(14,165,233,0.15); }
}
</style>
""", unsafe_allow_html=True)

# Header
hcol1, hcol2 = st.columns([4, 1])
with hcol1:
    st.markdown(
        "<h1 style='font-size:1.3rem;font-weight:700;margin:0 0 3px 0;'>"
        "Student AI Companion</h1>"
        "<p style='font-size:0.75rem;color:#6B7280;margin:0;'>"
        "M.Sc. Data Science &amp; Analytics &nbsp;&middot;&nbsp; "
        "Devi Ahilya Vishwavidyalaya, Indore</p>",
        unsafe_allow_html=True
    )
with hcol2:
    st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
    role = st.radio(
        "portal", ["Student", "Professor"],
        horizontal=True, label_visibility="collapsed", key="app_role"
    )

st.markdown(
    "<hr style='margin:1rem 0 1.6rem 0;border:none;"
    "border-top:1px solid #E5E7EB'>",
    unsafe_allow_html=True
)

if role == "Student":
    render_student_portal(syllabus)
else:
    render_professor_portal(syllabus)
