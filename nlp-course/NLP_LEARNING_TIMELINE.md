# NLP Course Learning Timeline & Roadmap

## Overview

This document provides detailed timelines for completing the 14-week Advanced NLP course at different paces. Choose the path that fits your schedule and goals.

---

## Quick Reference

| Path | Duration | Hours/Week | Best For |
|------|----------|------------|----------|
| **Intensive** | 14 weeks | 15-20 | Full-time students, career changers |
| **Accelerated** | 10 weeks | 20-25 | Experienced ML practitioners |
| **Part-Time** | 6 months | 8-10 | Working professionals |
| **Weekend** | 5 months | 10-12 | Weekend learners |

---

## Path 1: 14-Week Intensive (15-20 hours/week)

**Best for:** Full-time students, bootcamp participants, career changers

### Phase 1: Foundations (Weeks 1-4)

#### Week 1: Introduction & NLP Fundamentals
**Goal:** Understand NLP landscape and build first classifiers

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Read CMU 11-711 Lecture 1, Jurafsky Ch 1-2 | 4 |
| Wed-Thu | Set up environment, explore datasets | 4 |
| Fri-Sun | Implement rule-based & logistic classifier | 8 |

**Deliverable:** Working rule-based and statistical classifiers with comparison

---

#### Week 2: Word Representations & Text Classification
**Goal:** Master embeddings and build CNN/LSTM classifier

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study Word2Vec, GloVe papers | 5 |
| Wed-Thu | Implement SentencePiece tokenization | 5 |
| Fri-Sun | Build CNN text classifier, visualize embeddings | 8 |

**Deliverable:** CNN classifier with t-SNE embedding visualization

---

#### Week 3: Language Modeling
**Goal:** Understand n-gram and neural LMs

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Read Jurafsky LM chapter, implement n-gram | 6 |
| Wed-Thu | Study smoothing techniques, implement | 5 |
| Fri-Sun | Build feed-forward neural LM, compare perplexity | 7 |

**Deliverable:** N-gram LM with smoothing, neural LM comparison

---

#### Week 4: Sequence Modeling (RNNs, LSTMs)
**Goal:** Master recurrent architectures

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study RNN theory, vanishing gradients | 5 |
| Wed-Thu | Implement LSTM cell from scratch | 6 |
| Fri-Sun | Train LSTM LM, compare with Week 3 | 7 |

**Deliverable:** LSTM LM with lower perplexity than feed-forward

---

### Phase 2: Modern Architectures (Weeks 5-8)

#### Week 5: Transformers & Attention
**Goal:** Deep understanding of attention mechanisms

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Read "Attention Is All You Need" paper | 5 |
| Wed-Thu | Implement scaled dot-product & multi-head attention | 6 |
| Fri-Sun | Build Transformer classifier, compare with LSTM | 7 |

**Deliverable:** Transformer encoder with attention visualization

---

#### Week 6: Text Generation & Prompting
**Goal:** Master decoding strategies

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study decoding methods, nucleus sampling paper | 5 |
| Wed-Thu | Implement greedy, beam, top-k, nucleus sampling | 6 |
| Fri-Sun | Prompt engineering experiments | 6 |

**Deliverable:** Generation comparison across strategies

---

#### Week 7: Instruction Tuning & Efficient Fine-Tuning
**Goal:** Master PEFT methods

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study LoRA, adapters papers | 5 |
| Wed-Thu | Implement LoRA fine-tuning | 6 |
| Fri-Sun | Compare PEFT methods on QA task | 7 |

**Deliverable:** LoRA-tuned model with efficiency comparison

---

#### Week 8: Experimental Design & Annotation
**Goal:** Learn proper NLP methodology

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study experimental design, common pitfalls | 4 |
| Wed-Thu | Write annotation guidelines, annotate dataset | 6 |
| Fri-Sun | Calculate agreement metrics, write data statement | 5 |

**Deliverable:** Annotated dataset with Kappa scores and data statement

---

### Phase 3: Advanced Topics (Weeks 9-12)

#### Week 9: Retrieval & RAG
**Goal:** Build retrieval-augmented systems

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Implement BM25 from scratch | 5 |
| Wed-Thu | Build DPR system with FAISS | 6 |
| Fri-Sun | Complete RAG QA system | 8 |

**Deliverable:** Working RAG system with retrieval evaluation

---

#### Week 10: Distillation & RLHF
**Goal:** Understand model compression and alignment

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study distillation, quantization papers | 5 |
| Wed-Thu | Implement knowledge distillation | 6 |
| Fri-Sun | Train with DPO, compare with RLHF concepts | 7 |

**Deliverable:** Distilled model and DPO training experiment

---

#### Week 11: Debugging & Interpretation
**Goal:** Understand what models learn

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study probing, mechanistic interpretability | 5 |
| Wed-Thu | Implement probing classifiers | 6 |
| Fri-Sun | Attention analysis, model editing with ROME | 7 |

**Deliverable:** Probing analysis across BERT layers

---

#### Week 12: Advanced LLMs & Agents
**Goal:** Build LLM agents with tools

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study modern LLM architectures, RoPE | 5 |
| Wed-Thu | Implement ReAct agent framework | 7 |
| Fri-Sun | Build multi-tool agent, test | 8 |

**Deliverable:** ReAct agent with 3+ tools

---

### Phase 4: Specialized Topics (Weeks 13-14)

#### Week 13: Complex Reasoning
**Goal:** Master reasoning techniques

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Study chain-of-thought papers | 5 |
| Wed-Thu | Implement CoT, self-consistency | 6 |
| Fri-Sun | Compositional generalization experiments | 7 |

**Deliverable:** CoT implementation with self-consistency

---

#### Week 14: Multilingual NLP & Final Project
**Goal:** Cross-lingual transfer and course integration

| Day | Activity | Hours |
|-----|----------|-------|
| Mon-Tue | Zero-shot cross-lingual experiments | 6 |
| Wed-Thu | Start final project | 8 |
| Fri-Sun | Complete and document final project | 10 |

**Deliverable:** Final project integrating course concepts

---

## Path 2: 10-Week Accelerated (20-25 hours/week)

**Best for:** ML practitioners, researchers transitioning to NLP

### Weeks 1-2: Foundations
- Combine Weeks 1-4 content
- Focus on implementation over theory
- Skip detailed proofs, focus on intuition

### Weeks 3-4: Transformers & Generation
- Combine Weeks 5-6
- Deep dive into attention
- Master decoding strategies

### Weeks 5-6: Fine-tuning & Evaluation
- Combine Weeks 7-8
- LoRA implementation
- Proper experimental design

### Weeks 7-8: Advanced Topics
- Combine Weeks 9-11
- RAG systems
- RLHF/DPO basics

### Weeks 9-10: Modern LLMs & Wrap-up
- Combine Weeks 12-14
- Agents and reasoning
- Final project

---

## Path 3: 6-Month Part-Time (8-10 hours/week)

**Best for:** Working professionals, gradual learners

### Month 1: Foundations
- Weeks 1-3 content
- 2 weeks per course week
- Focus on understanding over speed

### Month 2: Sequence Modeling
- Weeks 4-6 content
- RNNs, Transformers, Generation
- Build working prototypes

### Month 3: Fine-tuning & Evaluation
- Weeks 7-8 content
- PEFT methods
- Experimental methodology

### Month 4: Retrieval & Alignment
- Weeks 9-10 content
- RAG systems
- RLHF basics

### Month 5: Interpretability & Agents
- Weeks 11-12 content
- Probing and debugging
- Agent frameworks

### Month 6: Reasoning & Multilingual
- Weeks 13-14 content
- Final project
- Course wrap-up

---

## Path 4: 5-Month Weekend Warrior (10-12 hours/weekend)

**Best for:** Full-time workers learning on weekends

### Month 1 (4 weekends)
- Weekend 1: Week 1 (Introduction)
- Weekend 2: Week 2 (Embeddings)
- Weekend 3: Week 3 (Language Models)
- Weekend 4: Week 4 (RNNs)

### Month 2 (4 weekends)
- Weekend 5: Week 5 (Transformers)
- Weekend 6: Week 6 (Generation)
- Weekend 7: Week 7 (Fine-tuning)
- Weekend 8: Week 8 (Evaluation)

### Month 3 (4 weekends)
- Weekend 9: Week 9 (RAG)
- Weekend 10: Week 10 (RLHF)
- Weekend 11: Week 11 (Interpretation)
- Weekend 12: Week 12 (Agents)

### Month 4 (4 weekends)
- Weekend 13: Week 13 (Reasoning)
- Weekend 14: Week 14 (Multilingual)
- Weekend 15-16: Final project

### Month 5 (if needed)
- Polish final project
- Review and consolidate
- Plan next learning steps

---

## Weekly Schedule Templates

### Intensive Learner (3+ hours/day)
```
Morning (1.5 hours):
- Paper reading / theory
- Take notes, highlight key points

Afternoon (2 hours):
- Coding exercises
- Implementation work

Evening (0.5 hours):
- Review day's learning
- Prepare for tomorrow
```

### Part-Time Professional (2 hours/day)
```
Weekday evenings (1 hour):
- Read papers, watch lectures
- Light coding exercises

Weekend (4-5 hours):
- Deep implementation work
- Project time
```

### Weekend Warrior (10-12 hours/weekend)
```
Saturday morning (3 hours):
- Theory and paper reading

Saturday afternoon (3 hours):
- Guided implementation

Sunday morning (3 hours):
- Independent coding

Sunday afternoon (3 hours):
- Debugging, documentation
```

---

## Checkpoints & Milestones

### Milestone 1: Foundation Complete (After Week 4)
- [ ] Can implement BoW and neural text classifier
- [ ] Understand word embeddings mathematically
- [ ] Can train and evaluate language models
- [ ] Implement LSTM from scratch

### Milestone 2: Modern NLP (After Week 8)
- [ ] Deep understanding of attention
- [ ] Can fine-tune with LoRA
- [ ] Understand decoding strategies
- [ ] Can design proper experiments

### Milestone 3: Advanced Practitioner (After Week 12)
- [ ] Can build RAG systems
- [ ] Understand RLHF pipeline
- [ ] Can interpret model behavior
- [ ] Can build LLM agents

### Milestone 4: Course Complete (After Week 14)
- [ ] Cross-lingual transfer experiments
- [ ] Chain-of-thought prompting mastery
- [ ] Completed substantial final project
- [ ] Ready for NLP roles

---

## Skills Assessment Rubric

Rate yourself 1-5 on each skill:

### Technical Implementation
- [ ] Text classification (BoW to Transformer): ___/5
- [ ] Language modeling (n-gram to neural): ___/5
- [ ] Attention mechanisms: ___/5
- [ ] Fine-tuning (full and PEFT): ___/5
- [ ] RAG systems: ___/5
- [ ] Agent development: ___/5

### Theoretical Understanding
- [ ] Word embeddings: ___/5
- [ ] Transformer architecture: ___/5
- [ ] RLHF/DPO: ___/5
- [ ] Interpretability: ___/5

### Research Skills
- [ ] Paper reading: ___/5
- [ ] Experiment design: ___/5
- [ ] Evaluation methodology: ___/5

**Target:** All skills at 4+ by end of course

---

## Adjustment Guidelines

### If You're Ahead of Schedule
- Implement papers not covered in course
- Build additional projects
- Contribute to open-source
- Start reading recent papers

### If You're Behind Schedule
- Focus on implementation over theory
- Use pre-built components where possible
- Skip optional readings
- Extend timeline without stress

### If You Need a Break
- Document where you stopped
- Save all work in progress
- Review basics when returning
- Don't restart from beginning

---

## Resource Time Allocation

Recommended split of study time:

```
Reading/Theory:     25% ████████
Implementation:     45% ██████████████
Projects:           20% ██████
Review/Debugging:   10% ███
```

---

## Final Checklist

By the end of your chosen path, you should be able to:

### Fundamentals
- [ ] Explain NLP evolution to a non-technical person
- [ ] Implement text classification from scratch
- [ ] Train and evaluate language models

### Modern Techniques
- [ ] Build Transformer-based systems
- [ ] Fine-tune efficiently with LoRA
- [ ] Apply proper experimental methodology

### Advanced Skills
- [ ] Build RAG systems
- [ ] Create LLM agents
- [ ] Apply chain-of-thought reasoning
- [ ] Work with multilingual models

### Professional Readiness
- [ ] 2+ substantial NLP projects
- [ ] Can read and implement papers
- [ ] Ready for NLP job interviews

---

## After Completion

### Next Steps
1. Build 2-3 portfolio projects
2. Write technical blog posts
3. Contribute to Hugging Face / open-source
4. Follow latest research
5. Specialize in area of interest

### Specialization Paths
- **Research**: Focus on papers, novel methods
- **Engineering**: Production systems, optimization
- **Applications**: Domain-specific NLP (bio, legal, finance)
- **Multimodal**: Vision + Language, Speech

---

**Remember:** This is a guide, not a strict rulebook. Adapt to your learning style and pace. The journey matters as much as the destination!

**Ready to start?** → [Begin with Week 1: Introduction](./week-01-introduction/README.md)
