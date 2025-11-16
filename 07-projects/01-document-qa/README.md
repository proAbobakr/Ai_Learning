# Project 1: Document Q&A System

## Project Overview

Build a complete question-answering system that can process multiple document formats and answer questions via a web interface.

**Difficulty:** Intermediate
**Time:** 8-12 hours
**Skills:** RAG, Web Development, Document Processing

## What You'll Build

A web application where users can:
1. Upload documents (PDF, DOCX, TXT)
2. Ask questions about the uploaded documents
3. Get answers with source citations
4. View chat history
5. Clear and re-upload documents

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Web Interface (Streamlit)          │
│  - File upload                                  │
│  - Question input                               │
│  - Answer display with citations                │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           Document Processing Layer             │
│  - PDF/DOCX/TXT parsing                         │
│  - Text chunking                                │
│  - Metadata extraction                          │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              RAG Engine                         │
│  - Embedding generation (OpenAI)                │
│  - Vector storage (ChromaDB)                    │
│  - Similarity search                            │
│  - LLM generation (GPT-3.5)                     │
└─────────────────────────────────────────────────┘
```

## Features

### Core Features
- ✅ Multi-format document upload (PDF, DOCX, TXT)
- ✅ Intelligent document chunking
- ✅ Question answering with context
- ✅ Source citation
- ✅ Chat history

### Advanced Features (Optional)
- 🔲 Multiple document support
- 🔲 Document search/filtering
- 🔲 Answer confidence scores
- 🔲 Export Q&A history
- 🔲 User authentication

## Prerequisites

```bash
pip install streamlit langchain openai chromadb pypdf2 python-docx
```

## Project Structure

```
01-document-qa/
├── README.md
├── app.py                 # Main Streamlit application
├── rag_engine.py          # RAG implementation
├── document_processor.py  # Document parsing and chunking
├── requirements.txt       # Dependencies
├── .env.example          # Example environment variables
└── data/                 # Uploaded documents (gitignored)
```

## Implementation Guide

### Step 1: Document Processor

**document_processor.py**
```python
"""
Document processing utilities for multiple file formats
"""

from typing import List, Dict
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process various document formats and chunk them"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF"""
        chunks = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                page_chunks = self.text_splitter.split_text(text)
                for chunk in page_chunks:
                    chunks.append({
                        'content': chunk,
                        'metadata': {
                            'source': file_path,
                            'page': page_num + 1
                        }
                    })
        return chunks

    def process_docx(self, file_path: str) -> List[Dict]:
        """Extract text from DOCX"""
        doc = Document(file_path)
        full_text = '\\n'.join([para.text for para in doc.paragraphs])
        text_chunks = self.text_splitter.split_text(full_text)
        chunks = [
            {
                'content': chunk,
                'metadata': {'source': file_path}
            }
            for chunk in text_chunks
        ]
        return chunks

    def process_txt(self, file_path: str) -> List[Dict]:
        """Extract text from TXT"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        text_chunks = self.text_splitter.split_text(text)
        chunks = [
            {
                'content': chunk,
                'metadata': {'source': file_path}
            }
            for chunk in text_chunks
        ]
        return chunks

    def process_file(self, file_path: str) -> List[Dict]:
        """Route to appropriate processor based on file extension"""
        if file_path.endswith('.pdf'):
            return self.process_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self.process_docx(file_path)
        elif file_path.endswith('.txt'):
            return self.process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
```

### Step 2: RAG Engine

**rag_engine.py**
```python
"""
RAG engine for question answering
"""

import os
from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from document_processor import DocumentProcessor


class RAGEngine:
    """RAG system for document question answering"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.vectorstore = None
        self.qa_chain = None
        self.doc_processor = DocumentProcessor()

    def load_documents(self, file_paths: List[str]):
        """Process and load documents into vector store"""
        all_chunks = []

        for file_path in file_paths:
            chunks = self.doc_processor.process_file(file_path)
            all_chunks.extend(chunks)

        # Create documents for vectorstore
        from langchain.schema import Document
        documents = [
            Document(page_content=chunk['content'], metadata=chunk['metadata'])
            for chunk in all_chunks
        ]

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        return len(all_chunks)

    def query(self, question: str) -> Dict:
        """Ask a question and get answer with sources"""
        if not self.qa_chain:
            raise ValueError("No documents loaded. Please upload documents first.")

        result = self.qa_chain({"query": question})

        return {
            'answer': result['result'],
            'sources': [
                {
                    'content': doc.page_content[:200] + '...',
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A')
                }
                for doc in result['source_documents']
            ]
        }

    def reset(self):
        """Clear the vector store and reset"""
        self.vectorstore = None
        self.qa_chain = None
        # Optionally delete persisted data
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
```

### Step 3: Streamlit Web Interface

**app.py**
```python
"""
Document Q&A Streamlit Application
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Initialize
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# Page config
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Document Q&A System")
st.markdown("Upload documents and ask questions!")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_dir = Path("./temp_uploads")
                temp_dir.mkdir(exist_ok=True)

                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(str(file_path))

                # Load into RAG engine
                num_chunks = st.session_state.rag_engine.load_documents(file_paths)

                st.session_state.documents_loaded = True
                st.success(f"✅ Processed {len(uploaded_files)} documents into {num_chunks} chunks!")

    if st.button("Clear Documents"):
        st.session_state.rag_engine.reset()
        st.session_state.chat_history = []
        st.session_state.documents_loaded = False
        st.success("Documents cleared!")

# Main area
if not st.session_state.documents_loaded:
    st.info("👈 Please upload and process documents to get started!")
else:
    # Question input
    question = st.text_input("❓ Ask a question about your documents:")

    if st.button("Get Answer") and question:
        with st.spinner("Searching for answer..."):
            try:
                result = st.session_state.rag_engine.query(question)

                # Add to history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources']
                })

            except Exception as e:
                st.error(f"Error: {e}")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("💬 Chat History")

        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question']}", expanded=(i==0)):
                st.markdown(f"**Answer:** {chat['answer']}")

                st.markdown("**Sources:**")
                for j, source in enumerate(chat['sources'], 1):
                    st.markdown(f"{j}. {source['source']} (Page {source['page']})")
                    st.caption(source['content'])
```

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "OPENAI_API_KEY=your-key" > .env

# Run the app
streamlit run app.py
```

## Testing

1. **Upload a document** (try the sample document from quickstart)
2. **Ask questions**:
   - "What are the main benefits?"
   - "How many vacation days are provided?"
   - "What is the remote work policy?"

## Enhancement Ideas

### Easy
- [ ] Add download button for chat history
- [ ] Show document statistics (pages, words)
- [ ] Add loading animations

### Medium
- [ ] Support for multiple document sets
- [ ] Search/filter documents
- [ ] Confidence scores for answers
- [ ] Export to PDF report

### Advanced
- [ ] User authentication
- [ ] Persistent storage (database)
- [ ] Admin dashboard
- [ ] API endpoint

## Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect your repo
4. Add OPENAI_API_KEY to secrets
5. Deploy!

### Deploy to Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
heroku config:set OPENAI_API_KEY=your-key
```

## Troubleshooting

### Issue: Can't process PDF
**Solution:** Check PDF isn't password-protected or corrupted

### Issue: Slow responses
**Solution:**
- Reduce chunk size
- Decrease k (retrieval count)
- Use GPT-3.5 instead of GPT-4

### Issue: Poor answers
**Solution:**
- Increase k (retrieve more context)
- Improve document quality
- Adjust chunk size/overlap

## Next Steps

- **Project 2:** [Customer Support Bot](../02-support-bot/README.md)
- **Project 3:** [Research Assistant](../03-research-assistant/README.md)

## Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [LangChain Docs](https://python.langchain.com/)
- [Project Demo Video](#) (coming soon)
