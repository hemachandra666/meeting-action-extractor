import json
from pathlib import Path
from time import perf_counter
from urllib.error import URLError

import streamlit as st

from meeting_extractor import extract_actions

st.set_page_config(page_title="Meeting action extractor")
st.title("Meeting action extractor")
st.caption("Hosted inference: submitted notes are sent to Groq (openai/gpt-oss-20b).")

sample_path = Path(__file__).with_name("hp-mixed-result.json")
sample = json.loads(sample_path.read_text(encoding="utf-8-sig"))

with st.form("extract"):
    note = st.text_area("Meeting note", value=sample["input_note"], height=150)
    submitted = st.form_submit_button("Extract actions")

if submitted:
    if note.strip() == sample["input_note"].strip():
        st.caption("Source: HP Q4 2025 earnings call, page 8. Elena’s follow-up is fictional.")
    if not note.strip():
        st.warning("Enter a meeting note first.")
    else:
        started = perf_counter()
        try:
            with st.spinner("Reading the note..."):
                actions = extract_actions(note)
        except (URLError, OSError, ValueError) as exc:
            st.error(f"Extraction failed: {exc}")
        else:
            elapsed = perf_counter() - started
            if actions:
                st.table(actions)
            else:
                st.info("No action items found.")
            st.caption(f"Hosted extraction time: {elapsed:.2f} seconds")

