import streamlit as st
from dotenv import load_dotenv
import os
import json

load_dotenv()

@st.cache_resource
def load_syllabus():
    syllabus_path = os.path.join(os.path.dirname(__file__), "syllabus.json")
    if os.path.exists(syllabus_path) and os.path.getsize(syllabus_path) > 0:
        with open(syllabus_path, "r") as f:
            return json.load(f)
    return {}

syllabus = load_syllabus()

st.set_page_config(
    page_title="Student AI Companion",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Student AI Companion")
    st.write("A knowledge base assistant for M.Sc. Data Science coursework.")
    st.divider()
    role = st.radio("Select your role", ["Student", "Professor"])

if role == "Student":
    st.header("Student Portal")
    st.info("Application modules are being built. Return here as each phase is completed.")

elif role == "Professor":
    st.header("Professor Portal")
    password = st.text_input("Enter professor password", type="password")
    prof_password = os.environ.get("PROFESSOR_PASSWORD", "")
    if password == prof_password and prof_password != "":
        st.success("Access granted.")
        st.info("Professor modules are being built.")
    elif password != "":
        st.error("Incorrect password.")
