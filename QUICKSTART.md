# Quick Start Guide - Build Your First RAG in 15 Minutes

## What You'll Build

A working question-answering system over your own documents using RAG.

## Prerequisites

- Python 3.8+ installed
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- 15 minutes

## Step 1: Setup (3 minutes)

```bash
# Clone or navigate to this repo
cd Ai_Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install langchain openai chromadb tiktoken python-dotenv
```

## Step 2: Configure API Key (1 minute)

Create `.env` file:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

Replace `your-key-here` with your actual OpenAI API key.

## Step 3: Create Sample Documents (2 minutes)

```bash
mkdir -p examples/quickstart/data
```

Create `examples/quickstart/data/knowledge.txt`:

```text
Our company offers the following benefits:

1. Health Insurance: Full medical, dental, and vision coverage. The company pays 80% of premiums.

2. Vacation: 15 days of paid vacation per year, plus 10 sick days.

3. Remote Work: Employees can work remotely up to 3 days per week.

4. 401(k): Company matches up to 5% of your salary in 401(k) contributions.

5. Professional Development: $2000 annual budget for courses, conferences, and certifications.
```

## Step 4: Create RAG Script (5 minutes)

Create `examples/quickstart/simple_rag.py`:

```python
import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

print("🚀 Initializing RAG system...\n")

# 1. Load documents
print("📄 Loading documents...")
loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"   Loaded {len(documents)} document(s)\n")

# 2. Split into chunks
print("✂️  Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"   Created {len(chunks)} chunks\n")

# 3. Create embeddings and vector store
print("🔢 Creating embeddings and vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
print("   Vector store created\n")

# 4. Create QA chain
print("🔗 Setting up QA chain...")
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)
print("   QA chain ready\n")

print("="*60)
print("✅ RAG System Ready! Ask your questions below.")
print("="*60 + "\n")

# 5. Ask questions
questions = [
    "What health insurance benefits do we offer?",
    "How many vacation days do employees get?",
    "What is the remote work policy?"
]

for question in questions:
    print(f"❓ Question: {question}")
    result = qa_chain({"query": question})
    print(f"💡 Answer: {result['result']}\n")
    print("-"*60 + "\n")
```

## Step 5: Run It! (4 minutes)

```bash
cd examples/quickstart
python simple_rag.py
```

**Expected Output:**

```
🚀 Initializing RAG system...

📄 Loading documents...
   Loaded 1 document(s)

✂️  Splitting into chunks...
   Created 2 chunks

🔢 Creating embeddings and vector store...
   Vector store created

🔗 Setting up QA chain...
   QA chain ready

============================================================
✅ RAG System Ready! Ask your questions below.
============================================================

❓ Question: What health insurance benefits do we offer?
💡 Answer: The company offers full medical, dental, and vision coverage, paying 80% of premiums.

------------------------------------------------------------

❓ Question: How many vacation days do employees get?
💡 Answer: Employees get 15 days of paid vacation per year, plus 10 sick days.

------------------------------------------------------------

❓ Question: What is the remote work policy?
💡 Answer: Employees can work remotely up to 3 days per week.

------------------------------------------------------------
```

## 🎉 Congratulations!

You just built your first RAG system!

## What Just Happened?

1. **Loaded** your text document
2. **Chunked** it into smaller pieces
3. **Embedded** chunks into vectors
4. **Stored** vectors in a database
5. **Retrieved** relevant chunks for each question
6. **Generated** answers using GPT

## Try This Next

### 1. Add Your Own Documents

```bash
# Add more .txt files to the data/ folder
echo "More content..." > data/additional_info.txt
python simple_rag.py
```

### 2. Ask Custom Questions

Modify the `questions` list in `simple_rag.py`:

```python
questions = [
    "Your question here",
    "Another question",
]
```

### 3. Make It Interactive

Add this at the end of `simple_rag.py`:

```python
# Interactive mode
print("\n💬 Interactive Mode (type 'quit' to exit)\n")
while True:
    question = input("Your question: ")
    if question.lower() in ['quit', 'exit', 'q']:
        break
    result = qa_chain({"query": question})
    print(f"Answer: {result['result']}\n")
```

### 4. Adjust Parameters

Experiment with these:

```python
# More chunks retrieved
retriever=vectorstore.as_retriever(search_kwargs={"k": 5})

# Smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

# More creative answers
llm = OpenAI(temperature=0.7)
```

## Understanding the Cost

For this quickstart:
- **Embeddings**: ~$0.0001 (negligible)
- **3 Questions**: ~$0.003 (less than a penny)
- **Total**: < $0.01

## Troubleshooting

### Error: "No API key found"
```bash
# Make sure .env file exists and contains:
OPENAI_API_KEY=sk-your-actual-key
```

### Error: "No module named 'langchain'"
```bash
pip install langchain openai chromadb tiktoken python-dotenv
```

### Poor/Wrong Answers
- Add more context to your documents
- Increase `k` to retrieve more chunks
- Try different chunk sizes

## Next Steps

Ready to go deeper? Choose your path:

1. **Quick Learner?** → [3-Month Accelerated Path](./LEARNING_TIMELINE.md#path-2-3-month-accelerated-10-15-hoursweek)

2. **Want Foundations?** → [Module 1: Prerequisites](./01-prerequisites/README.md)

3. **Hands-on Focused?** → [Module 3: Basic Implementation](./03-basic-implementation/01-simple-rag.md)

4. **Structured Learning?** → [Complete Course Overview](./README.md)

## Resources

- **Documentation**: [LangChain Docs](https://python.langchain.com/)
- **Community**: [LangChain Discord](https://discord.gg/langchain)
- **Examples**: Check `07-projects/` for full applications

## Share Your Success!

Built your first RAG? Share it:
- Twitter: Tag #RAG #AI
- LinkedIn: Show your learning journey
- GitHub: Fork and add your experiments

---

**Questions?** Open an issue or discussion in this repo.

**Happy Building! 🚀**
