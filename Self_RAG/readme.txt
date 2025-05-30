# Self RAG Application

## Overview
This is a simple Retrieval-Augmented Generation (RAG) app where users can:
- Upload PDF documents
- Ask questions about the document
- Receive AI-generated answers using Groq LLMs

## Setup

1. Clone the repo and navigate to the project folder.

2. Install dependencies:
   pip install -r requirements.txt

3. Add your Groq API key in a `.env` file:
   GROQ_API_KEY=your_groq_api_key_here

4. Run the app:
   streamlit run app.py

## Notes
- Uses FAISS for fast vector search
- HuggingFace embeddings: all-MiniLM-L6-v2
- LLM: Mixtral from Groq (low-latency, high-quality)
