import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

GROQ_API_KEY="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"

#Load_Docs
def load_docs(file_path):
    loader=PyPDFLoader(file_path)
    documents= loader.load()
    return documents

#Create vector,text spliter and embedding

def create_vectorstore(docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks=splitter.split_documents(docs)
    
    embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore=FAISS.from_documents(chunks,embeddings)
    return vectorstore

#call LLM
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192"
    )

def get_rag_chain(vectorstore): #similary search
    retriever=vectorstore.as_retriever(search_kwargs={"k":3}) # k=3 best high 1st 3
    llm=get_llm()
    rag_chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    return rag_chain