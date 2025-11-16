# Video Tutorial Scripts & Documentation

## Overview

This directory contains scripts, outlines, and supplementary materials for video tutorials covering the entire AI & LLM course.

**Total Videos**: 50+
**Total Duration**: ~100 hours
**Format**: Screen recording + code walkthrough

---

## Video Series Structure

### Series 1: Python & Math Foundations (10 videos, ~20 hours)

#### Video 1.1: Python Essentials for AI (2 hours)
**Script**: [01_python_essentials.md](./scripts/01_python_essentials.md)

**Outline**:
1. **Introduction** (5 min)
   - Why Python for AI?
   - Course overview
   - Setup verification

2. **Data Structures** (30 min)
   - Lists for datasets
   - Dictionaries for configurations
   - Sets for unique elements
   - Live coding examples

3. **List Comprehensions** (20 min)
   - Basic syntax
   - Filtering and transforming
   - Nested comprehensions
   - Real ML use cases

4. **Functions** (25 min)
   - Function basics
   - Lambda expressions
   - Decorators for timing
   - Example: normalization function

5. **Object-Oriented Programming** (30 min)
   - Classes for ML models
   - Inheritance
   - Example: Linear regression class

6. **Practice Exercise** (10 min)
   - Build data preprocessing class
   - Live coding

**Code Files**:
- `examples/01_python_essentials.py`
- `exercises/python_exercises.py`

**Key Takeaways**:
- Python is the language of AI/ML
- List comprehensions are essential
- OOP helps organize code
- Practice daily!

---

#### Video 1.2: NumPy Mastery (2.5 hours)
**Script**: [02_numpy_mastery.md](./scripts/02_numpy_mastery.md)

**Outline**:
1. **Why NumPy?** (10 min)
   - Speed comparison vs Python lists
   - Vectorization benefits
   - Memory efficiency

2. **Array Creation** (30 min)
   - 15+ creation methods
   - When to use each
   - Live demonstrations

3. **Array Operations** (40 min)
   - Element-wise operations
   - Matrix multiplication
   - Broadcasting (critical!)
   - Real examples

4. **Indexing & Slicing** (30 min)
   - Boolean masking
   - Fancy indexing
   - Practical use cases

5. **Linear Algebra** (40 min)
   - Matrix operations
   - Eigenvalues/vectors
   - Applications in ML

6. **Practice** (20 min)
   - Build neural network forward pass
   - Implement gradient descent

**Demo**: Build simple neural network using only NumPy

---

#### Video 1.3: Linear Algebra Visualized (2 hours)

**Outline**:
1. **Vectors Fundamentals** (30 min)
   - Vector representation
   - Vector addition/subtraction
   - Scalar multiplication
   - Visualizations

2. **Matrices** (40 min)
   - Matrix operations
   - Matrix multiplication
   - Why order matters
   - Neural network connections

3. **Eigenvalues & Eigenvectors** (30 min)
   - What they mean
   - How to compute
   - PCA application

4. **Applications** (20 min)
   - Dimensionality reduction
   - Data transformation
   - Neural network layers

**Visualizations**: 20+ interactive plots

---

### Series 2: Machine Learning (12 videos, ~25 hours)

#### Video 2.1: Linear Regression from Scratch (2.5 hours)

**Outline**:
1. **Theory** (20 min)
   - What is regression?
   - Cost function
   - Gradient descent

2. **Implementation** (60 min)
   - Code from scratch
   - No libraries
   - Line by line explanation

3. **Real Dataset** (40 min)
   - Load data
   - Preprocess
   - Train model
   - Evaluate

4. **Visualizations** (20 min)
   - Fitted line
   - Loss curve
   - Residuals

5. **Q&A** (10 min)

**Code**: Complete implementation in `examples/`

---

#### Video 2.2: Logistic Regression Deep Dive (2.5 hours)

**Topics**:
- Sigmoid function
- Cross-entropy loss
- Binary classification
- Multi-class with softmax
- Real dataset (Iris, Cancer)

---

#### Video 2.3: Decision Trees & Random Forests (3 hours)

**Topics**:
- Decision tree algorithm
- Entropy and information gain
- Building from scratch
- Random forest ensemble
- Feature importance

---

### Series 3: Deep Learning (15 videos, ~30 hours)

#### Video 3.1: Neural Networks from Scratch (3 hours)

**The Big One!**

**Outline**:
1. **Perceptron** (30 min)
2. **Multi-layer networks** (40 min)
3. **Backpropagation** (60 min)
4. **Training loop** (40 min)
5. **MNIST example** (50 min)

**Deliverable**: Working neural network, no frameworks

---

#### Video 3.2: CNNs for Image Classification (3 hours)

**Topics**:
- Convolution operation
- Pooling layers
- CNN architectures
- Transfer learning
- Image classification project

---

#### Video 3.3: RNNs and LSTMs (2.5 hours)

**Topics**:
- Sequence modeling
- RNN architecture
- LSTM cells
- Text generation
- Time series prediction

---

#### Video 3.4: Transformers Explained (4 hours)

**Detailed Walkthrough**:

**Part 1: Attention Mechanism** (60 min)
- Query, Key, Value
- Attention scores
- Scaled dot-product attention
- Code implementation

**Part 2: Multi-Head Attention** (45 min)
- Multiple attention heads
- Why it works
- Implementation

**Part 3: Transformer Block** (60 min)
- Feed-forward network
- Layer normalization
- Residual connections
- Complete block

**Part 4: Full Transformer** (45 min)
- Encoder stack
- Decoder stack
- Training loop

**Part 5: Applications** (30 min)
- Language modeling
- Translation
- Modern variants

---

### Series 4: Large Language Models (12 videos, ~25 hours)

#### Video 4.1: GPT Architecture Deep Dive (3 hours)

**Contents**:
1. **History** (20 min)
   - GPT-1, GPT-2, GPT-3
   - Architecture evolution
   - Key innovations

2. **Architecture** (60 min)
   - Decoder-only transformer
   - Positional encodings
   - Attention patterns
   - Implementation

3. **Pre-training** (40 min)
   - Causal language modeling
   - Training objectives
   - Data requirements

4. **Hands-On** (60 min)
   - Build GPT-2 from scratch
   - Train on small dataset
   - Generate text

---

#### Video 4.2: Prompt Engineering Masterclass (2.5 hours)

**100+ Prompt Examples!**

**Outline**:
1. **Basics** (30 min)
   - What is a prompt?
   - Zero-shot vs few-shot
   - Best practices

2. **Advanced Techniques** (60 min)
   - Chain-of-thought
   - ReAct prompting
   - Tree of thoughts
   - 30+ examples

3. **Prompt Optimization** (40 min)
   - A/B testing prompts
   - Systematic improvement
   - Evaluation

4. **Real Applications** (20 min)
   - Customer support
   - Code generation
   - Data analysis

---

#### Video 4.3: RAG Implementation (3 hours)

**Complete Tutorial**:
1. Setup vector database
2. Document processing
3. Embedding creation
4. Retrieval system
5. LLM integration
6. Full application

---

### Series 5: Fine-tuning LLMs (10 videos, ~20 hours)

#### Video 5.1: LoRA Fine-tuning Explained (2.5 hours)

**Outline**:
1. **Theory** (40 min)
   - Why LoRA?
   - Low-rank decomposition
   - Mathematics explained

2. **Implementation** (60 min)
   - Install PEFT
   - Configure LoRA
   - Fine-tune LLaMA-2

3. **Evaluation** (30 min)
   - Test model
   - Compare to base
   - Metrics

4. **Production** (20 min)
   - Save/load model
   - Deployment tips

---

#### Video 5.2: QLoRA for 4-bit Fine-tuning (2 hours)

**Topics**:
- 4-bit quantization
- QLoRA configuration
- Training 7B model on single GPU
- Memory optimization

---

#### Video 5.3: RLHF Tutorial (3 hours)

**Complete RLHF Pipeline**:
1. Reward model training
2. PPO implementation
3. DPO alternative
4. Alignment techniques

---

## Video Production Guidelines

### Recording Setup
- **Screen Resolution**: 1920x1080
- **Recording Software**: OBS Studio
- **Code Editor**: VS Code with Monokai theme
- **Terminal**: iTerm2 or Windows Terminal
- **Font Size**: 16pt minimum

### Video Structure
1. **Intro** (2-3 min)
   - Topic overview
   - Learning objectives
   - Prerequisites

2. **Theory** (20-30%)
   - Concepts explained
   - Visualizations
   - Intuition building

3. **Code Walkthrough** (50-60%)
   - Live coding
   - Line-by-line explanation
   - Common mistakes

4. **Practice** (15-20%)
   - Exercise
   - Solution
   - Tips

5. **Outro** (2-3 min)
   - Summary
   - Next steps
   - Resources

### Editing Checklist
- [ ] Remove long pauses
- [ ] Add chapter markers
- [ ] Include code timestamps
- [ ] Add captions/subtitles
- [ ] Insert graphics/diagrams
- [ ] Include download links

---

## Supplementary Materials

### For Each Video

1. **Slides** (PDF)
   - Key concepts
   - Diagrams
   - Code snippets

2. **Code** (GitHub)
   - Complete working code
   - Exercise templates
   - Solutions

3. **Notes** (Markdown)
   - Written summary
   - Additional resources
   - Further reading

4. **Quiz** (10 questions)
   - Test understanding
   - Auto-graded

---

## Video Hosting

### Platforms
- **YouTube**: Main channel (public)
- **Vimeo**: High quality (members)
- **Course Platform**: Integrated player

### Organization
- Playlists by module
- Sequential numbering
- Consistent thumbnails
- SEO-optimized titles

---

## Community Features

### Live Coding Sessions
- **Weekly**: 2 hours
- **Q&A**: Live questions
- **Recordings**: Available after

### Office Hours
- **Bi-weekly**: 1 hour
- **Help**: Individual problems
- **Recordings**: Available

---

## Video Timeline

### Month 1: Foundation Videos
- Python & NumPy (4 videos)
- Math foundations (6 videos)

### Month 2: ML Videos
- Classical ML (12 videos)

### Month 3: Deep Learning
- Neural networks (8 videos)
- CNNs, RNNs (7 videos)

### Month 4: NLP & Transformers
- NLP basics (5 videos)
- Transformers (7 videos)

### Month 5-6: LLMs & Fine-tuning
- LLM fundamentals (8 videos)
- Fine-tuning (10 videos)
- Advanced topics (6 videos)

---

## Production Stats

**Total Videos**: 50+
**Total Duration**: ~100 hours
**Total Code**: 10,000+ lines
**Total Slides**: 500+ pages

**Time to Produce**: 6 months
**Update Frequency**: Quarterly

---

## Next Steps

To actually create these videos:
1. Record screen with code
2. Edit with DaVinci Resolve / Final Cut
3. Add captions and graphics
4. Upload to YouTube
5. Create playlists
6. Promote in community

---

**Note**: This documentation provides the structure and scripts. Actual video recording would be done separately with screen recording software.

**For now**: All code examples are available to run locally. These scripts serve as detailed guides for self-paced learning.

---

*Last Updated: 2025-11-16*
*Videos: In Development*
*Scripts: Complete*
