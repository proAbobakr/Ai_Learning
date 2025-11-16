# Building Your First RAG System

## Overview

In this tutorial, you'll build a complete, working RAG system from scratch. We'll start simple and progressively add features.

**What you'll build:**
A question-answering system over a collection of text documents.

**Time required:** 30-45 minutes

---

## Prerequisites

```bash
# Install required packages
pip install langchain openai chromadb tiktoken
```

```bash
# Set up environment variables
export OPENAI_API_KEY='your-api-key-here'
```

Or create `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

---

## Project Structure

```
simple-rag/
├── data/
│   ├── doc1.txt
│   ├── doc2.txt
│   └── doc3.txt
├── .env
├── rag_system.py
└── query.py
```

---

## Step 1: Prepare Sample Documents

Create some sample documents to work with.

**data/doc1.txt**
```text
Company Vacation Policy

All full-time employees are entitled to 15 days of paid vacation per year.
Vacation days accrue at a rate of 1.25 days per month.
Employees must request vacation at least 2 weeks in advance.
Unused vacation days can be carried over to the next year, up to a maximum of 5 days.
Vacation requests are subject to manager approval based on team coverage needs.
```

**data/doc2.txt**
```text
Remote Work Policy

Employees may work remotely up to 3 days per week.
Remote work requires manager approval and must be scheduled in advance.
All remote employees must be available during core hours (10 AM - 3 PM EST).
A stable internet connection and home office setup are required.
VPN access is mandatory when working remotely.
Remote employees must attend all scheduled team meetings via video conference.
```

**data/doc3.txt**
```text
Health Insurance Benefits

The company offers comprehensive health insurance coverage for all full-time employees.
Coverage begins on the first day of employment.
The company pays 80% of the premium, employees pay 20%.
Coverage includes medical, dental, and vision insurance.
Employees can add family members to their plan at additional cost.
Open enrollment period is every November for the following year.
```

---

## Step 2: Build Basic RAG System

**rag_system.py**

```python
"""
Simple RAG System Implementation
Author: Your Name
"""

import os
from typing import List
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


class SimpleRAG:
    """A simple RAG system for question answering over documents"""

    def __init__(self, data_dir: str, persist_dir: str = "./chroma_db"):
        """
        Initialize the RAG system

        Args:
            data_dir: Directory containing text documents
            persist_dir: Directory to store the vector database
        """
        load_dotenv()

        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.qa_chain = None

        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)

    def load_documents(self) -> List:
        """Load all text documents from data directory"""
        print(f"Loading documents from {self.data_dir}...")

        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        return documents

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        print("Splitting documents into chunks...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        return chunks

    def create_vectorstore(self, chunks: List):
        """Create and persist vector store"""
        print("Creating vector store...")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        self.vectorstore.persist()
        print(f"Vector store created and saved to {self.persist_dir}")

    def load_vectorstore(self):
        """Load existing vector store"""
        print(f"Loading vector store from {self.persist_dir}...")

        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

        print("Vector store loaded")

    def setup_qa_chain(self):
        """Set up the question-answering chain"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore() or load_vectorstore() first.")

        print("Setting up QA chain...")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 chunks
            ),
            return_source_documents=True
        )

        print("QA chain ready")

    def query(self, question: str) -> dict:
        """
        Ask a question and get an answer

        Args:
            question: The question to ask

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")

        print(f"\nQuestion: {question}")
        print("Searching for relevant information...")

        result = self.qa_chain({"query": question})

        # Format response
        response = {
            "question": question,
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                }
                for doc in result["source_documents"]
            ]
        }

        return response

    def initialize(self, use_existing: bool = False):
        """
        Complete initialization workflow

        Args:
            use_existing: If True, load existing vector store. If False, create new one.
        """
        if use_existing and os.path.exists(self.persist_dir):
            self.load_vectorstore()
        else:
            # Load and process documents
            documents = self.load_documents()
            chunks = self.split_documents(documents)
            self.create_vectorstore(chunks)

        # Set up QA chain
        self.setup_qa_chain()

        print("\n✓ RAG system initialized and ready!\n")


def main():
    """Example usage"""

    # Create RAG system
    rag = SimpleRAG(data_dir="./data")

    # Initialize (set use_existing=True to load existing DB)
    rag.initialize(use_existing=False)

    # Ask questions
    questions = [
        "How many vacation days do employees get?",
        "What is the remote work policy?",
        "When does health insurance coverage begin?",
    ]

    for question in questions:
        result = rag.query(question)

        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources used: {len(result['sources'])}")
        for i, source in enumerate(result['sources'], 1):
            print(f"\nSource {i}:")
            print(f"File: {source['source']}")
            print(f"Content: {source['content'][:100]}...")

        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
```

---

## Step 3: Run the System

```bash
# Create data directory with sample documents
mkdir -p data

# Copy the sample documents above to data/doc1.txt, doc2.txt, doc3.txt

# Run the RAG system
python rag_system.py
```

**Expected Output:**
```
Loading documents from ./data...
Loaded 3 documents
Splitting documents into chunks...
Created 6 chunks
Creating vector store...
Vector store created and saved to ./chroma_db

✓ RAG system initialized and ready!

Question: How many vacation days do employees get?
Searching for relevant information...

Answer: All full-time employees are entitled to 15 days of paid vacation per year.

Sources used: 3

Source 1:
File: data/doc1.txt
Content: Company Vacation Policy

All full-time employees are entitled to 15 days of paid vacation per...

================================================================================
```

---

## Step 4: Interactive Query Script

**query.py** - For interactive querying

```python
"""Interactive query interface for RAG system"""

from rag_system import SimpleRAG


def interactive_query():
    """Interactive command-line interface"""

    # Initialize RAG system
    print("Initializing RAG system...")
    rag = SimpleRAG(data_dir="./data")
    rag.initialize(use_existing=True)  # Load existing DB

    print("\n" + "="*80)
    print("RAG Question Answering System")
    print("="*80)
    print("\nAsk questions about the company policies.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        # Get question from user
        question = input("Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not question:
            continue

        try:
            # Get answer
            result = rag.query(question)

            # Display answer
            print(f"\n{'Answer:':<10} {result['answer']}\n")

            # Display sources
            print(f"{'Sources:':<10}")
            for i, source in enumerate(result['sources'], 1):
                file_name = source['source'].split('/')[-1]
                print(f"  [{i}] {file_name}")

            print()

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    interactive_query()
```

**Usage:**
```bash
python query.py
```

**Example Session:**
```
Your question: Can I work from home?

Answer:     Yes, employees may work remotely up to 3 days per week, subject to
            manager approval and scheduling in advance.

Sources:
  [1] doc2.txt
  [2] doc2.txt

Your question: How much does health insurance cost?

Answer:     The company pays 80% of the premium and employees pay 20%.

Sources:
  [1] doc3.txt

Your question: quit

Goodbye!
```

---

## How It Works

### 1. Document Loading
```python
DirectoryLoader → Finds all .txt files in data/
TextLoader → Reads each file's content
Result: List of Document objects
```

### 2. Chunking
```python
RecursiveCharacterTextSplitter
    ↓ chunk_size=500 (characters)
    ↓ chunk_overlap=50 (maintain context)
Result: Smaller, manageable chunks
```

### 3. Embedding & Storage
```python
OpenAIEmbeddings → Convert chunks to vectors
ChromaDB → Store vectors with metadata
Result: Searchable vector database
```

### 4. Query Processing
```python
User Question
    ↓ OpenAIEmbeddings
Query Vector
    ↓ ChromaDB similarity_search(k=3)
Top 3 Relevant Chunks
    ↓ Combine into prompt
LLM (OpenAI)
    ↓
Answer
```

---

## Experimentation Ideas

Try modifying these parameters and observe the results:

### 1. Chunk Size
```python
# rag_system.py, line ~52
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Try: 200, 500, 1000
    chunk_overlap=50,
)
```

### 2. Number of Retrieved Chunks
```python
# rag_system.py, line ~97
retriever=self.vectorstore.as_retriever(
    search_kwargs={"k": 5}  # Try: 1, 3, 5, 10
)
```

### 3. LLM Temperature
```python
# rag_system.py, line ~36
self.llm = OpenAI(temperature=0.3)  # Try: 0, 0.5, 0.7
```

### 4. Retrieval Type
```python
# Try different retrieval methods
retriever=self.vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)
```

---

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError
```
Solution: pip install langchain openai chromadb
```

### Issue 2: OpenAI API Error
```
Solution: Check your API key in .env file
Ensure OPENAI_API_KEY is set correctly
```

### Issue 3: Empty or Poor Answers
```
Solutions:
- Increase k (retrieve more chunks)
- Reduce chunk_size (more precise chunks)
- Add more/better source documents
- Check if question is answerable from documents
```

### Issue 4: Slow Performance
```
Solutions:
- Use use_existing=True to reuse vector store
- Reduce k (fewer chunks to process)
- Use GPT-3.5 instead of GPT-4
```

---

## Cost Estimation

For this simple example:

**One-time Indexing:**
- 3 documents → ~6 chunks
- Embedding cost: ~$0.0001 (negligible)

**Per Query:**
- Query embedding: $0.0001
- LLM call (3 chunks + query): ~$0.001-0.003

**Total: < $0.01 for entire tutorial**

---

## Next Steps

Now that you have a working RAG system, you can:

1. **Add more documents**: Try PDFs, Word docs, web pages
2. **Improve chunking**: Learn advanced strategies
3. **Add a UI**: Build a Streamlit or Gradio interface
4. **Optimize retrieval**: Hybrid search, re-ranking
5. **Deploy**: Make it accessible to others

**[→ Document Processing](./02-document-processing.md)**

Learn to handle:
- PDFs
- Word documents
- Web pages
- Code files
- Structured data
