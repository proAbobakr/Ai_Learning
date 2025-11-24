# AI Learning Hub

Welcome to the comprehensive AI/ML learning repository! This hub contains multiple courses designed to take you from beginner to advanced practitioner in various AI domains.

---

## Available Courses

### 1. RAG (Retrieval-Augmented Generation) - Zero to Hero
| Duration | Level | Focus |
|----------|-------|-------|
| 10 weeks | Beginner to Advanced | Building RAG systems |

Build RAG systems from scratch, covering vector databases, embeddings, retrieval strategies, and production deployment.

**Quick Links:** [Course Content](#rag-course-content) | [Quick Start](./QUICKSTART.md) | [Timeline](./LEARNING_TIMELINE.md)

---

### 2. Advanced NLP - From Fundamentals to Frontier Models
| Duration | Level | Focus |
|----------|-------|-------|
| 14 weeks | Intermediate to Advanced | Comprehensive NLP |

Comprehensive NLP curriculum covering everything from word embeddings to modern LLMs, transformers, RLHF, agents, and multilingual NLP. Based on CMU 11-711 and UMass CS 685.

**Topics Covered:**
- Word Representations & Language Modeling
- RNNs, LSTMs, Transformers & Attention
- Text Generation & Prompt Engineering
- Instruction Tuning & LoRA/PEFT
- Retrieval-Augmented Generation
- RLHF & Direct Preference Optimization
- Mechanistic Interpretability
- LLM Agents & Tool Use
- Chain-of-Thought Reasoning
- Multilingual NLP & Cross-lingual Transfer

**Quick Links:** [Start NLP Course](./nlp-course/README.md) | [NLP Timeline](./nlp-course/NLP_LEARNING_TIMELINE.md)

---

## Course Comparison

| Aspect | RAG Course | NLP Course |
|--------|------------|------------|
| **Duration** | 10 weeks | 14 weeks |
| **Prerequisites** | Python basics, ML fundamentals | Python, ML basics, Linear Algebra |
| **Focus** | Production RAG systems | Theoretical + Practical NLP |
| **Best For** | Engineers building RAG apps | Researchers, ML Engineers |
| **Key Outcome** | Deploy RAG systems | Full NLP understanding |

---

# RAG Course Content

## RAG (Retrieval-Augmented Generation) - Zero to Hero Course

This course takes you from complete beginner to advanced practitioner in building RAG systems.

## 📚 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI framework that combines the power of large language models (LLMs) with external knowledge retrieval. Instead of relying solely on the knowledge embedded in the model's parameters, RAG systems fetch relevant information from external sources to generate more accurate, up-to-date, and contextually relevant responses.

## 🎯 Course Overview

This course is structured as a progressive learning path:

1. **Prerequisites** - Foundation knowledge needed
2. **RAG Fundamentals** - Core concepts and architecture
3. **Basic Implementation** - Build your first RAG system
4. **Intermediate Techniques** - Vector databases and embeddings
5. **Advanced RAG** - Optimization and production-ready systems
6. **Hands-on Projects** - Real-world applications

## 📖 Course Structure

### Module 1: Prerequisites (Week 1)
- [Prerequisites Guide](./01-prerequisites/README.md)
  - Python programming basics
  - Machine learning fundamentals
  - Natural language processing concepts
  - Required libraries and tools

### Module 2: RAG Fundamentals (Week 1-2)
- [Understanding RAG](./02-fundamentals/01-what-is-rag.md)
- [RAG Architecture](./02-fundamentals/02-architecture.md)
- [Key Components](./02-fundamentals/03-components.md)
- [Use Cases](./02-fundamentals/04-use-cases.md)

### Module 3: Basic RAG Implementation (Week 2-3)
- [Simple RAG with OpenAI](./03-basic-implementation/01-simple-rag.md)
- [Document Processing](./03-basic-implementation/02-document-processing.md)
- [Text Chunking Strategies](./03-basic-implementation/03-chunking.md)
- [Basic Retrieval](./03-basic-implementation/04-retrieval.md)

### Module 4: Intermediate RAG (Week 3-5)
- [Vector Embeddings Deep Dive](./04-intermediate/01-embeddings.md)
- [Vector Databases (Pinecone, Weaviate, ChromaDB)](./04-intermediate/02-vector-databases.md)
- [Similarity Search](./04-intermediate/03-similarity-search.md)
- [Complete RAG Pipeline](./04-intermediate/04-complete-pipeline.md)

### Module 5: Advanced RAG Techniques (Week 5-7)
- [Hybrid Search](./05-advanced/01-hybrid-search.md)
- [Re-ranking Strategies](./05-advanced/02-reranking.md)
- [Query Optimization](./05-advanced/03-query-optimization.md)
- [Context Window Management](./05-advanced/04-context-management.md)
- [Multi-modal RAG](./05-advanced/05-multimodal.md)

### Module 6: Production & Optimization (Week 7-8)
- [Performance Optimization](./06-production/01-optimization.md)
- [Evaluation Metrics](./06-production/02-evaluation.md)
- [Deployment Strategies](./06-production/03-deployment.md)
- [Monitoring and Maintenance](./06-production/04-monitoring.md)

### Module 7: Hands-on Projects (Week 8-10)
- [Project 1: Document Q&A System](./07-projects/01-document-qa/README.md)
- [Project 2: Customer Support Bot](./07-projects/02-support-bot/README.md)
- [Project 3: Research Assistant](./07-projects/03-research-assistant/README.md)
- [Project 4: Code Documentation Helper](./07-projects/04-code-helper/README.md)

## ⏱️ Learning Timeline

### **10-Week Intensive Path**
- **Weeks 1-2**: Prerequisites & Fundamentals
- **Weeks 2-3**: Basic Implementation
- **Weeks 3-5**: Intermediate Techniques
- **Weeks 5-7**: Advanced Methods
- **Weeks 7-8**: Production Skills
- **Weeks 8-10**: Projects & Practice

### **6-Month Part-Time Path**
- **Month 1**: Prerequisites & Fundamentals
- **Month 2**: Basic & Intermediate Implementation
- **Month 3**: Advanced Techniques
- **Month 4**: Production & Optimization
- **Months 5-6**: Projects & Real-world Applications

### **3-Month Accelerated Path** (10-15 hours/week)
- **Month 1**: Prerequisites through Basic Implementation
- **Month 2**: Intermediate & Advanced Techniques
- **Month 3**: Production Skills & Projects

## 🛠️ Tools & Technologies Used

- **Python 3.8+**
- **LangChain** - RAG framework
- **OpenAI API** / **Hugging Face** - LLMs
- **Vector Databases**: ChromaDB, Pinecone, Weaviate, FAISS
- **Embeddings**: OpenAI, Sentence Transformers
- **Document Loaders**: PyPDF2, python-docx, BeautifulSoup
- **Web Frameworks**: FastAPI, Streamlit (for demos)

## 🎓 Learning Outcomes

By completing this course, you will be able to:

✅ Understand the fundamentals of RAG architecture
✅ Build basic to advanced RAG systems
✅ Implement efficient document processing pipelines
✅ Work with vector databases and embeddings
✅ Optimize RAG systems for production
✅ Evaluate and improve RAG performance
✅ Deploy RAG applications at scale

## 📋 Prerequisites

Before starting this course, you should have:

- **Python**: Intermediate proficiency (functions, classes, decorators)
- **Basic ML/AI**: Understanding of neural networks and transformers
- **API Knowledge**: Experience with REST APIs
- **Command Line**: Comfortable with terminal/bash
- **Optional**: Familiarity with Docker and cloud platforms

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   # Clone this repository
   git clone <your-repo-url>
   cd Ai_Learning

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **API Keys Setup**
   - OpenAI API key (for GPT models)
   - Pinecone API key (for vector database)
   - Hugging Face token (optional, for open-source models)

3. **Start with Module 1**
   - Begin with [Prerequisites Guide](./01-prerequisites/README.md)
   - Follow the modules in order
   - Complete exercises and projects

## 📚 Additional Resources

- **Documentation**: Links to official docs for all tools
- **Research Papers**: Key papers on RAG and retrieval systems
- **Community**: Discord/Slack channels for questions
- **Updates**: This course is continuously updated

## 🤝 Contributing

Found an error or want to add content? Contributions are welcome!

## 📄 License

This course material is provided for educational purposes.

---

**Ready to begin?** Start with [Module 1: Prerequisites](./01-prerequisites/README.md)
