import streamlit as st
from rag_self import process_query

st.title("🧠 Self RAG - Ask Your Docs")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF Uploaded Successfully")

    user_question = st.text_input("Enter your question:")
    if user_question:
        with st.spinner("Processing..."):
            answer = process_query(uploaded_file, user_question)
        st.markdown(f"**Answer:** {answer}")
