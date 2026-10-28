"""Streamlit web app for the College Study Assistant.

Run with:  streamlit run app.py
"""

import os

# Windows fix: PyTorch and FAISS each bring their own OpenMP library, and loading
# both can crash the whole server. These must be set BEFORE importing them.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from embedder import ModelLoadError, warm_up  # noqa: E402
from rag_pipeline import StudyAssistant  # noqa: E402
from llm_client import api_key_available  # noqa: E402

st.set_page_config(page_title="College Study Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (first time takes a minute)...")
def get_assistant() -> StudyAssistant:
    warm_up()  # load PyTorch and the model once, here, with the spinner showing
    return StudyAssistant()


try:
    assistant = get_assistant()
except ModelLoadError as err:
    st.error(str(err))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📂 Your notes")
    uploads = st.file_uploader(
        "Upload PDF, TXT or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", type="primary", disabled=not uploads):
        for f in uploads:
            bar = st.progress(0.0, text=f"Reading {f.name}...")
            try:
                count = assistant.add_document(
                    f.getvalue(),
                    f.name,
                    on_progress=lambda done, total, n=f.name: bar.progress(
                        done / total, text=f"{n}: {done}/{total} chunks"
                    ),
                )
                bar.empty()
                st.success(f"{f.name}: {count} chunks")
            except ValueError as err:
                bar.empty()
                st.error(str(err))

    if assistant.files:
        st.subheader("Indexed files")
        for name, count in list(assistant.files.items()):
            col1, col2 = st.columns([4, 1])
            col1.write(f"{name} ({count} chunks)")
            if col2.button("✖", key=f"rm_{name}", help="Remove this file"):
                assistant.remove_document(name)
                st.rerun()

    st.divider()
    top_k = st.slider("Passages to use", 1, 8, 4)
    if api_key_available():
        st.caption(f"✅ LLM connected ({os.getenv('GEMINI_MODEL', 'gemini-flash-latest')})")
    else:
        st.caption(":warning: No GEMINI_API_KEY found. The app will show raw passages only.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main page ----------
st.title("📚 College Study Assistant")
st.caption("Upload your notes, then ask questions. Answers come from your own documents.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s.file}**, page {s.page} (match {s.score:.2f})")
                    st.caption(s.preview)

question = st.chat_input("Ask something, e.g. Explain binary search trees with an example")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            answer = assistant.ask(question, k=top_k)
        st.markdown(answer.text)
        if answer.warning:
            st.warning(f"LLM problem, showing passages instead. ({answer.warning})")
        if answer.sources:
            with st.expander("Sources"):
                for s in answer.sources:
                    st.markdown(f"**{s.file}**, page {s.page} (match {s.score:.2f})")
                    st.caption(s.preview)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "sources": answer.sources}
    )
