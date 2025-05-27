Corrective RAG Application (LangChain + Streamlit)

📌 Description:
This is a Retrieval-Augmented Generation (RAG) application that allows users to:
- Upload a PDF
- Ask questions based on its content
- Receive accurate, AI-generated answers using Groq + LangChain + FAISS

📁 Files:
- app.py: Streamlit frontend
- rag_corrective.py: Backend logic (PDF loader, vector DB, QA chain)
- requirements.txt: Python packages
- .env: Add your Groq API key here

⚙️ Setup Instructions:
1. Clone this repo or copy all files to a folder
2. Install the requirements:
   pip install -r requirements.txt

3. Create a `.env` file and add:
   GROQ_API_KEY=your_groq_key_here

4. Run the app:
   streamlit run app.py

🚀 Tech Stack:
- 🧠 LLM: LLaMA 3 via Groq API (Free tier model)
- 🔍 Retriever: FAISS (offline, fast, vector-based search)
- 📎 Embeddings: HuggingFace sentence-transformers (`all-MiniLM-L6-v2`)
- 📄 Document Splitter: RecursiveChunker for better context
- 🧾 PDF Parsing: PyPDFLoader

💡 Notes:
- This app is optimized for fast question answering on PDF content.
- Embedding and vector indexing happens on every new upload.
- Recommended: switch to persistent FAISS storage for faster re-use.

🌐 Deployment:
You can deploy this app to:
- Streamlit Community Cloud (free)
- Hugging Face Spaces (streamlit)
- Render / Railway / Fly.io

