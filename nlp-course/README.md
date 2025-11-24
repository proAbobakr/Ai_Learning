# Advanced NLP - From Fundamentals to Frontier Models

Welcome to the comprehensive 14-week Natural Language Processing course! This curriculum takes you from NLP fundamentals through modern large language models, covering both theoretical foundations and practical implementations.

## Course Overview

This course is structured around cutting-edge NLP education, drawing from leading university curricula including CMU 11-711 and UMass CS 685. It provides a rigorous yet practical path through modern NLP.

### What You'll Learn

- **Foundations**: History of NLP, word representations, and language modeling
- **Neural Architectures**: RNNs, LSTMs, Transformers, and attention mechanisms
- **Modern LLMs**: GPT, BERT, T5, LLaMA, and instruction-tuned models
- **Advanced Techniques**: RLHF, RAG, model compression, and interpretability
- **Practical Skills**: From simple classifiers to production-ready systems

## Course Structure

### Part 1: Foundations (Weeks 1-4)

| Week | Topic | Key Focus |
|------|-------|-----------|
| [Week 1](./week-01-introduction/README.md) | Introduction & NLP Fundamentals | History, tasks, rule-based vs. neural |
| [Week 2](./week-02-word-representations/README.md) | Word Representations & Classification | Embeddings, word2vec, GloVe, text classification |
| [Week 3](./week-03-language-modeling/README.md) | Language Modeling | N-grams, neural LMs, perplexity |
| [Week 4](./week-04-sequence-modeling/README.md) | Sequence Modeling | RNNs, LSTMs, GRUs, vanishing gradients |

### Part 2: Modern Architectures (Weeks 5-8)

| Week | Topic | Key Focus |
|------|-------|-----------|
| [Week 5](./week-05-transformers/README.md) | Transformers & Attention | Self-attention, multi-head attention, positional encoding |
| [Week 6](./week-06-text-generation/README.md) | Text Generation & Prompting | Decoding strategies, prompt engineering |
| [Week 7](./week-07-instruction-tuning/README.md) | Instruction Tuning & Fine-Tuning | LoRA, adapters, FLAN, few-shot learning |
| [Week 8](./week-08-experimental-design/README.md) | Experimental Design & Annotation | Evaluation, data collection, inter-annotator agreement |

### Part 3: Advanced Topics (Weeks 9-12)

| Week | Topic | Key Focus |
|------|-------|-----------|
| [Week 9](./week-09-retrieval-rag/README.md) | Retrieval & RAG | BM25, DPR, retrieval-augmented generation |
| [Week 10](./week-10-distillation-rlhf/README.md) | Distillation & RLHF | Model compression, RLHF, DPO |
| [Week 11](./week-11-debugging-interpretation/README.md) | Debugging & Interpretation | Probing, mechanistic interpretability, ROME |
| [Week 12](./week-12-advanced-llms/README.md) | Advanced LLMs & Agents | LLaMA, GPT-4, tool use, long contexts |

### Part 4: Specialized Topics (Weeks 13-14)

| Week | Topic | Key Focus |
|------|-------|-----------|
| [Week 13](./week-13-complex-reasoning/README.md) | Complex Reasoning & Linguistics | Chain-of-thought, compositional generalization |
| [Week 14](./week-14-multilingual/README.md) | Multilingual NLP & Wrap-Up | mBERT, XLM-R, cross-lingual transfer, final project |

## Learning Paths

### Intensive Path (14 weeks, 15-20 hours/week)
Follow the curriculum week by week, completing all readings, exercises, and projects.

### Accelerated Path (10 weeks, 20+ hours/week)
- Weeks 1-2: Foundations (Course Weeks 1-4)
- Weeks 3-4: Transformers & Generation (Course Weeks 5-6)
- Weeks 5-6: Fine-tuning & Evaluation (Course Weeks 7-8)
- Weeks 7-8: Advanced Topics (Course Weeks 9-11)
- Weeks 9-10: LLMs, Reasoning & Multilingual (Course Weeks 12-14)

### Part-Time Path (6 months, 8-10 hours/week)
- Month 1: Weeks 1-3 (Foundations & Language Modeling)
- Month 2: Weeks 4-6 (Sequences, Transformers, Generation)
- Month 3: Weeks 7-8 (Fine-tuning & Evaluation)
- Month 4: Weeks 9-10 (RAG & Model Compression)
- Month 5: Weeks 11-12 (Interpretability & Modern LLMs)
- Month 6: Weeks 13-14 (Reasoning, Multilingual & Projects)

## Prerequisites

Before starting this course, you should have:

- **Python**: Intermediate proficiency (NumPy, pandas, OOP)
- **Machine Learning**: Basic understanding of supervised learning, gradient descent
- **Linear Algebra**: Vectors, matrices, matrix multiplication
- **Calculus**: Derivatives, chain rule (for understanding backpropagation)
- **Probability**: Basic probability, conditional probability

### Recommended Preparation
- Complete a basic ML course (Andrew Ng's ML course, fast.ai)
- Familiarity with PyTorch or TensorFlow basics
- Experience with Jupyter notebooks

## Tools & Technologies

### Core Frameworks
- **PyTorch** - Primary deep learning framework
- **Hugging Face Transformers** - Pre-trained models and tokenizers
- **Hugging Face Datasets** - Standard NLP datasets

### Additional Libraries
- **SentencePiece** - Subword tokenization
- **FAISS** - Vector similarity search
- **Weights & Biases** - Experiment tracking
- **spaCy** - NLP preprocessing

### Computing Requirements
- GPU access recommended (Google Colab, Kaggle, or local GPU)
- Minimum 16GB RAM for larger models
- ~50GB disk space for models and datasets

## Key References

This course draws from foundational papers and resources:

### Foundational Papers
- Vaswani et al. (2017) - "Attention Is All You Need"
- Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"
- Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)
- Ouyang et al. (2022) - "Training language models to follow instructions with human feedback"

### University Courses
- **CMU 11-711**: Advanced NLP
- **UMass CS 685**: Advanced NLP
- **Stanford CS224N**: NLP with Deep Learning

### Textbooks
- Jurafsky & Martin - "Speech and Language Processing" (3rd ed.)
- Goldberg - "Neural Network Methods for Natural Language Processing"

## Learning Outcomes

By completing this course, you will be able to:

- Understand the evolution of NLP from rule-based to neural approaches
- Implement text classification, sequence modeling, and generation systems
- Work with modern transformer architectures and pre-trained models
- Fine-tune large language models using parameter-efficient methods
- Build retrieval-augmented generation (RAG) systems
- Apply RLHF and preference optimization techniques
- Interpret and debug neural NLP models
- Design NLP experiments with proper evaluation methodology
- Work with multilingual models and cross-lingual transfer

## Getting Started

1. **Environment Setup**
   ```bash
   # Clone the repository
   git clone <your-repo-url>
   cd Ai_Learning/nlp-course

   # Create virtual environment
   python -m venv nlp-env
   source nlp-env/bin/activate  # Windows: nlp-env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Verify Setup**
   ```python
   import torch
   import transformers
   print(f"PyTorch: {torch.__version__}")
   print(f"Transformers: {transformers.__version__}")
   print(f"GPU available: {torch.cuda.is_available()}")
   ```

3. **Begin Learning**
   - Start with [Week 1: Introduction & NLP Fundamentals](./week-01-introduction/README.md)
   - Follow the weekly structure
   - Complete exercises before moving on

## Weekly Structure

Each week includes:

1. **README.md** - Overview, core concepts, learning objectives
2. **Key References** - Papers, book chapters, blog posts to read
3. **Exercises** - Hands-on coding assignments
4. **Additional Resources** - Optional deep-dives

### Recommended Weekly Schedule

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Read papers and theory | 4-6 |
| Wed-Thu | Work through tutorials | 4-6 |
| Fri-Sun | Complete exercises | 6-8 |

## Assessment & Progress

### Self-Assessment Checkpoints

**After Week 4 (Foundations):**
- [ ] Can explain word2vec and GloVe mathematically
- [ ] Can implement an LSTM from scratch
- [ ] Understand perplexity and language model evaluation

**After Week 8 (Modern Architectures):**
- [ ] Can implement transformer attention mechanisms
- [ ] Understand different decoding strategies
- [ ] Can fine-tune models with LoRA

**After Week 12 (Advanced Topics):**
- [ ] Can build a RAG system
- [ ] Understand RLHF pipeline
- [ ] Can probe and interpret model behaviors

**After Week 14 (Complete):**
- [ ] Can work with multilingual models
- [ ] Can apply chain-of-thought prompting
- [ ] Have completed a substantial NLP project

## Contributing

Found an error or want to suggest improvements? Contributions are welcome!

## License

This course material is provided for educational purposes.

---

**Ready to begin?** Start with [Week 1: Introduction & NLP Fundamentals](./week-01-introduction/README.md)
