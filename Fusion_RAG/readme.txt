Fusion RAG - PDF Question Answering with Hybrid Retrieval and LLM
=================================================================

Overview:
---------
Fusion RAG is an interactive Streamlit application that allows users to upload PDF documents and ask questions about the content. The system uses a hybrid retrieval approach combining vector similarity search with multi-query retrievers, powered by Langchain, Groq LLM (llama3-8b-8192), and HuggingFace embeddings.

Features:
---------
- Upload PDF files for on-the-fly document processing.
- Split documents into optimized chunks for efficient retrieval.
- Create vector embeddings using HuggingFace sentence transformers.
- Hybrid retrieval using multi-query retriever to improve context coverage.
- Generate answers using Groq's llama3-8b-8192 LLM.
- Interactive Q&A interface with Streamlit.

How to Use:
-----------
1. Clone the repository:
   git clone https://github.com/your-username/fusion-rag.git
   cd fusion-rag

2. Install dependencies:
   pip install -r requirements.txt

3. Set your Groq API key:
   Create a `.env` file in the root directory with:
   GROQ_API_KEY=your_actual_groq_api_key

4. Run the app:
   streamlit run app.py

5. Upload a PDF file and ask your questions directly.

Project Structure:
------------------
- app.py: Main Streamlit application.
- fusion_rag.py: Core functions for loading PDFs, vector store creation, hybrid retriever, and QA chain.
- requirements.txt: Required Python packages.
- .env: Environment variables (not included, user must create).

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
- huggingface-hub
- torch
- sentence-transformers

Notes:
------
- The multi-query retriever enhances retrieval by generating multiple relevant queries for each user input.
- The LLM prompt encourages the model to answer based only on provided context and decline to answer if unsure.
- Make sure the Groq API key is valid and set in your environment variables.

Contributing:
-------------
Contributions are welcome! Feel free to submit issues and pull requests to improve this project.

License:
--------
MIT License
