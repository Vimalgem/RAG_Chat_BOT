# app.py

import streamlit as st
import tempfile
import os
from rag_speculative import load_pdf, create_vectorstore, get_qa_chain

st.set_page_config(page_title="🧠 Speculative RAG", layout="centered")

st.title("📄 Speculative RAG - Ask Questions from Your PDF")
st.markdown("Upload a PDF and ask questions powered by **LLM + RAG**.")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# PDF Upload Section
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("🔄 Processing PDF... Please wait."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile_path = tmpfile.name

        docs = load_pdf(tmpfile_path)
        vectorstore = create_vectorstore(docs)
        st.session_state.qa_chain = get_qa_chain(vectorstore)

        os.unlink(tmpfile_path)  # Remove temporary file
        st.success("✅ PDF processed successfully!")

# Question & Answer Section
if st.session_state.qa_chain:
    user_query = st.text_input("❓ Ask a question about the document:")

    if user_query:
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.qa_chain.run(user_query)
            st.markdown("### 💡 Answer:")
            st.write(response)