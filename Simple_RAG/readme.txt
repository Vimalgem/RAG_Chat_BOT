RAG Chatbot with Langchain and Groq
===================================

Description:
-------------
This project is a Retrieval-Augmented Generation (RAG) Chatbot built using Streamlit for the frontend interface, Langchain for document processing and vector search, and Groq LLM for generating answers. It allows users to upload PDF documents, processes and indexes them, and then lets users ask questions based on the document content.

Features:
---------
- Upload PDF files and automatically extract and split the content.
- Create vector embeddings of the document chunks for efficient retrieval.
- Use Groq LLM (llama3-8b-8192 model) to generate context-aware answers.
- Display source document snippets related to the generated answer.
- Interactive UI using Streamlit.

Setup Instructions:
-------------------
1. Clone the repository:
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

2. Install required dependencies:
   pip install -r requirements.txt

3. Set your Groq API key:
   Replace "your_apikey" in rag_chain.py with your actual Groq API key or use environment variables.

4. Run the Streamlit app:
   streamlit run app.py

Usage:
------
1. Open the Streamlit app in your browser.
2. Upload a PDF file using the uploader.
3. Wait for the document to be processed and indexed.
4. Enter your question in the text input box.
5. View the generated answer and explore source chunks for context.

File Structure:
---------------
- app.py: Main Streamlit app file.
- rag_chain.py: Contains functions for loading documents, creating vector stores, calling Groq LLM, and assembling the RAG chain.
- requirements.txt: List of Python packages required.

Dependencies:
-------------
- langchain
- langchain-community
- langchainhub
- streamlit
- faiss-cpu
- groq
- python-dotenv
- langchain-groq

Notes:
------
- Ensure you have a valid Groq API key for the ChatGroq LLM.
- The current text splitter uses chunk size 1000 with overlap 100 for balanced chunking.
- The retrieval step returns the top 3 relevant chunks for answering.

Contributing:
-------------
Feel free to fork this repo and submit pull requests for improvements, bug fixes, or new features.

License:
--------
MIT License



