"""
Simple RAG System - Quickstart Example
Build a question-answering system in 15 minutes
"""

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

def main():
    print("🚀 Initializing RAG system...\n")

    # 1. Load documents
    print("📄 Loading documents...")
    loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"   Loaded {len(documents)} document(s)\n")

    # 2. Split into chunks
    print("✂️  Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
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

    # 5. Ask predefined questions
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

    # 6. Interactive mode
    print("\n💬 Interactive Mode (type 'quit' to exit)\n")
    while True:
        question = input("Your question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not question.strip():
            continue

        try:
            result = qa_chain({"query": question})
            print(f"💡 Answer: {result['result']}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
