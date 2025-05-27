#main code
import streamlit as st
import tempfile
from rag_chain import load_docs, create_vectorstore,get_rag_chain

st.set_page_config(page_title="Rag Chatbot",layout="centered")
st.title("RAG Chatbot with langchain and groq")


uploaded_file=st.file_uploader("Upload a file",type=['pdf'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path=temp_file.name
        
    st.success("PDF Uploaded Successfully")
    
    with st.spinner("Processing and indexing the documents"):
        documents=load_docs(temp_file_path)
        vectorstore=create_vectorstore(documents)
        rag_chain=get_rag_chain(vectorstore)
    
    st.success("Document loaded and Ask your question now")
    
    #User question
    User_question=st.text_input("Ask your questions")
    
    if User_question:
        with st.spinner("Generating response"):
            response=rag_chain({"query":User_question})
            answer=response['result']       # answer stored in result
            sources=response['source_documents']
        st.subheader("Answer")
        st.write(answer)
        
        #Upload file show
        with st.expander("source chunks"):
            for i, docs in enumerate(sources):
                st.markdown(f"**source{i+1}**")
                st.write(docs.page_content[:500]+"...")
else:
    st.info("Please upload a file")
    
    
    