import streamlit as st
import tempfile
from rag_corrective import process_pdf, get_qa_chain


st.set_page_config(page_title="Corrective RAG", layout="centered")
st.title("📄 Corrective RAG - Ask Questions from Your PDF")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file and "qa_chain" not in st.session_state:
    with st.spinner("🔄 Processing PDF and generating embeddings..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        retriever = process_pdf(tmp_path)
        st.session_state.qa_chain = get_qa_chain(retriever)

    st.success("✅ PDF processed. You can now ask questions.")

# Show question box only after processing
if "qa_chain" in st.session_state:
    query = st.text_input("🧠 Ask a question based on the uploaded PDF:")
    if query:
        with st.spinner("🤖 Generating answer..."):
            result = st.session_state.qa_chain.run(query)
            st.markdown(f"**Answer:** {result}")
