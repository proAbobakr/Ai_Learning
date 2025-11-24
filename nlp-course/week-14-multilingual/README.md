# Week 14: Multilingual NLP & Course Wrap-Up

## Overview

This final week covers multilingual NLP: how to build models that work across languages, transfer learning from high-resource to low-resource languages, and the challenges of linguistic diversity. We conclude with a capstone project integrating knowledge from the entire course.

## Learning Objectives

By the end of this week, you will be able to:
- Understand multilingual embedding spaces
- Fine-tune multilingual models (mBERT, XLM-R)
- Perform zero-shot cross-lingual transfer
- Recognize challenges in multilingual NLP
- Integrate course knowledge into a final project
- Identify areas for continued learning

## Core Topics

### 1. The Multilingual Challenge

**Scale of the Problem**:
- ~7,000 languages worldwide
- NLP resources exist for ~100-200
- Most models trained primarily on English

**Resource Distribution**:
```
English:        ████████████████████  (massive)
Chinese/Spanish: ███████████          (large)
German/French:   █████████            (medium)
Swahili/Welsh:   ███                  (small)
Most languages:  █                    (minimal/none)
```

**Why Multilingual?**:
- Most of the world doesn't speak English
- Preserve linguistic diversity
- Business/communication needs
- Scientific knowledge in many languages

### 2. Multilingual Word Embeddings

**Cross-lingual Embedding Spaces**:
```
English space:         Aligned space:
  cat•                   cat• gato• 猫•
  dog•                   dog• perro• 犬•
        →
Spanish space:
  gato•
  perro•
```

**Alignment Methods**:

**Supervised (with dictionary)**:
```python
# Learn mapping W such that WX_en ≈ X_es
# For word pairs (en_word, es_word) in dictionary
def learn_mapping(X_en, X_es, pairs):
    # Procrustes solution
    X_en_paired = X_en[pairs[:, 0]]
    X_es_paired = X_es[pairs[:, 1]]

    U, _, Vt = np.linalg.svd(X_es_paired.T @ X_en_paired)
    W = U @ Vt
    return W
```

**Unsupervised**:
- Adversarial training to match distributions
- No bilingual dictionary needed
- Works surprisingly well

**Joint Training**:
- Train on multilingual corpora
- Shared vocabulary (subwords)
- Natural alignment emerges

### 3. Multilingual BERT (mBERT)

**Architecture**:
- Same as BERT (12 layers, 768 hidden)
- Trained on 104 languages
- Shared WordPiece vocabulary (110K tokens)

**Training**:
- Wikipedia dumps from all languages
- Same MLM objective
- No explicit alignment signal

**Surprising Finding**:
- Cross-lingual representations emerge
- Zero-shot transfer works
- Fine-tune on English → works on other languages

### 4. XLM-RoBERTa (XLM-R)

**Improvements over mBERT**:
- RoBERTa training (more data, longer)
- 100 languages
- CommonCrawl data (not just Wikipedia)
- Better low-resource performance

```python
from transformers import XLMRobertaModel, XLMRobertaTokenizer

model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Works on any supported language
text_en = "Hello, how are you?"
text_fr = "Bonjour, comment allez-vous?"
text_zh = "你好，你好吗？"
```

### 5. Zero-Shot Cross-Lingual Transfer

**The Setup**:
```
Training: English labeled data
Testing:  Spanish/German/Chinese (no labels)
```

**How It Works**:
1. Fine-tune XLM-R on English task
2. Apply directly to other languages
3. Shared representation enables transfer

```python
from transformers import XLMRobertaForSequenceClassification

# Fine-tune on English sentiment
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')
train(model, english_sentiment_data)

# Zero-shot inference on Spanish
spanish_predictions = model.predict(spanish_text)
```

**Performance**:
- Often 70-90% of supervised performance
- Better for similar languages (EN→DE vs EN→ZH)
- Degrades for low-resource languages

### 6. Multilingual Machine Translation

**Google's Multilingual NMT** (Johnson et al., 2016):
```
Input:  <2es> Hello, how are you?
Output: Hola, ¿cómo estás?

Input:  <2fr> Hello, how are you?
Output: Bonjour, comment allez-vous?
```

**Benefits**:
- One model for many language pairs
- Transfer to low-resource pairs
- Zero-shot translation (X→Y without X-Y data)

**NLLB - No Language Left Behind** (2022):
- 200 languages
- Focuses on low-resource languages
- Significant improvements for underserved languages

### 7. Challenges in Multilingual NLP

**The Curse of Multilinguality**:
- Fixed capacity shared across languages
- Adding languages can hurt high-resource performance
- Need to balance coverage vs. depth

**Script and Tokenization**:
- Different scripts (Latin, Cyrillic, CJK, Arabic)
- Tokenization challenges (no spaces in Chinese)
- Subword vocabulary allocation

**Linguistic Diversity**:
- Word order (SVO, SOV, VSO)
- Morphology (isolating vs. agglutinative)
- Grammatical gender, case systems

**Evaluation Challenges**:
- Benchmarks mostly in English
- Translation-based evaluation biased
- Need native speaker evaluation

### 8. Language-Specific Adaptations

**When Zero-Shot Isn't Enough**:

**Few-Shot Cross-Lingual**:
```python
# Mix some target language data
train_data = english_data + small_spanish_data
model = train(XLMRoberta, train_data)
```

**Translate-Train**:
```python
# Translate English data to target language
spanish_train = translate(english_train, target='es')
model = train(XLMRoberta, spanish_train)
```

**Translate-Test**:
```python
# Translate target language to English at inference
english_text = translate(spanish_text, target='en')
prediction = english_model.predict(english_text)
```

**Language-Adaptive Pre-training**:
- Continue pre-training on target language
- Adapts representations without labeled data

### 9. Low-Resource NLP

**Strategies**:
1. **Transfer from related languages**
2. **Data augmentation** (back-translation)
3. **Cross-lingual projection**
4. **Active learning** (select most informative examples)

**AfroLM, AfriBERTa**:
- Models for African languages
- Address neglected language families
- Community-driven efforts

### 10. Course Integration: Final Project

**Project Option 1: Multilingual RAG System**
```
Build a RAG system that:
1. Retrieves from documents in multiple languages
2. Answers questions in the user's language
3. Uses XLM-R for retrieval, LLM for generation
```

**Project Option 2: Cross-Lingual Sentiment Analysis**
```
Build a system that:
1. Fine-tunes on English sentiment data
2. Transfers to 3+ other languages
3. Compares zero-shot vs few-shot performance
```

**Project Option 3: Instruction-Tuned Multilingual Model**
```
Build a system that:
1. Takes a small open model (Mistral, LLaMA)
2. Instruction-tunes with multilingual data
3. Evaluates on cross-lingual benchmarks
```

## Key References

### Required Reading

1. **Johnson et al. (2016) - "Google's Multilingual Neural Machine Translation System"**
   - Multilingual NMT with language tags
   - [Paper](https://arxiv.org/abs/1611.04558)

2. **Conneau et al. (2020) - "Unsupervised Cross-lingual Representation Learning at Scale"**
   - XLM-RoBERTa
   - [Paper](https://arxiv.org/abs/1911.02116)

3. **NLLB Team (2022) - "No Language Left Behind"**
   - 200-language translation
   - [Paper](https://arxiv.org/abs/2207.04672)

### Recommended Reading

4. **Pires et al. (2019) - "How Multilingual is Multilingual BERT?"**
   - Analysis of mBERT's cross-lingual abilities
   - [Paper](https://arxiv.org/abs/1906.01502)

5. **Wu & Dredze (2019) - "Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT"**
   - Spanish BERT and cross-lingual transfer
   - [Paper](https://arxiv.org/abs/1904.09077)

6. **Artetxe et al. (2020) - "A Call for More Rigor in Unsupervised Cross-lingual Learning"**
   - Evaluation best practices
   - [Paper](https://arxiv.org/abs/2004.14958)

## Practical Exercise

### Exercise 1: Zero-Shot Cross-Lingual Transfer

**Objective**: Fine-tune on English, evaluate on multiple languages

```python
from transformers import XLMRobertaForSequenceClassification
from datasets import load_dataset

# Load multilingual sentiment dataset
dataset = load_dataset("amazon_reviews_multi")

# Fine-tune on English
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')
train(model, dataset['en'])

# Evaluate on other languages
results = {}
for lang in ['de', 'es', 'fr', 'ja', 'zh']:
    results[lang] = evaluate(model, dataset[lang])

print(results)
```

### Exercise 2: Multilingual Embedding Analysis

**Objective**: Visualize cross-lingual alignment

```python
from transformers import XLMRobertaModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Get embeddings for translation pairs
words = {
    'en': ['dog', 'cat', 'house', 'water', 'tree'],
    'es': ['perro', 'gato', 'casa', 'agua', 'árbol'],
    'de': ['Hund', 'Katze', 'Haus', 'Wasser', 'Baum'],
}

embeddings = {}
for lang, word_list in words.items():
    embeddings[lang] = [get_word_embedding(w, model) for w in word_list]

# Visualize alignment
# Plot and analyze clustering
```

### Exercise 3: Final Project

**Choose one project from Section 10 and implement**

**Requirements**:
1. Use techniques from at least 4 weeks of the course
2. Include proper evaluation
3. Document methodology and results
4. Discuss limitations and future work

**Suggested Components**:
- Week 2: Tokenization
- Week 5: Transformers
- Week 7: Fine-tuning
- Week 9: Retrieval (if RAG)
- Week 14: Multilingual

### Exercise Files
- [exercises/cross_lingual_transfer.py](./exercises/cross_lingual_transfer.py)
- [exercises/multilingual_embeddings.py](./exercises/multilingual_embeddings.py)
- [exercises/final_project_template/](./exercises/final_project_template/)

## Course Summary

### Key Takeaways by Week

| Week | Topic | Key Concept |
|------|-------|-------------|
| 1 | Introduction | NLP evolution: rules → statistics → neural |
| 2 | Representations | Word embeddings capture meaning |
| 3 | Language Models | Predict next word, measure with perplexity |
| 4 | Sequence Models | LSTMs solve vanishing gradient |
| 5 | Transformers | Attention enables parallelization |
| 6 | Generation | Decoding strategies matter |
| 7 | Fine-tuning | PEFT enables efficient adaptation |
| 8 | Evaluation | Rigorous methodology is essential |
| 9 | Retrieval | RAG grounds LLMs in knowledge |
| 10 | Alignment | RLHF/DPO align models to humans |
| 11 | Interpretation | Understand what models learn |
| 12 | Agents | LLMs + tools = powerful systems |
| 13 | Reasoning | Chain-of-thought improves reasoning |
| 14 | Multilingual | Cross-lingual transfer extends reach |

### Skills Acquired

**Implementation**:
- [ ] Build text classifiers (BoW → Transformer)
- [ ] Train language models (n-gram → neural)
- [ ] Fine-tune with LoRA/PEFT
- [ ] Build RAG systems
- [ ] Create LLM agents

**Analysis**:
- [ ] Evaluate models properly
- [ ] Interpret model behavior
- [ ] Debug failures
- [ ] Measure annotation agreement

**Research**:
- [ ] Read and understand NLP papers
- [ ] Replicate key results
- [ ] Design experiments
- [ ] Identify limitations

## Study Questions

1. Why does cross-lingual transfer work in multilingual models?
2. What causes the "curse of multilinguality"?
3. How do you choose between zero-shot, translate-train, and translate-test?
4. What are the ethical considerations in multilingual NLP?
5. How would you approach NLP for a truly low-resource language?

## Continued Learning

### Areas to Explore

1. **Multimodal**: Vision + Language (CLIP, GPT-4V)
2. **Speech**: ASR, TTS, Speech Translation
3. **Code**: Code generation, program synthesis
4. **Scientific NLP**: BioNLP, Legal NLP, etc.
5. **Efficiency**: Smaller models, edge deployment

### Resources

**Courses**:
- CMU 11-711: Continues to be updated
- Stanford CS224N: NLP with Deep Learning
- Fast.ai NLP course

**Conferences**:
- ACL, EMNLP, NAACL (NLP)
- NeurIPS, ICML, ICLR (ML)

**Communities**:
- Hugging Face forums
- r/MachineLearning
- NLP Twitter/Mastodon

**Staying Current**:
- arXiv cs.CL daily
- Papers with Code
- Research lab blogs (Google AI, Meta AI, Anthropic)

## Final Checklist

**Technical Skills**:
- [ ] Can implement models from papers
- [ ] Can evaluate properly with baselines
- [ ] Can fine-tune efficiently
- [ ] Can build end-to-end systems

**Knowledge**:
- [ ] Understand attention and transformers deeply
- [ ] Know when to use which technique
- [ ] Can read and critique NLP papers
- [ ] Aware of limitations and failure modes

**Practical**:
- [ ] Built 2+ substantial NLP projects
- [ ] Can explain concepts to others
- [ ] Ready for NLP job interviews
- [ ] Know where to learn more

---

## Congratulations!

You've completed a comprehensive journey through modern NLP. You now have the foundation to:

- Build production NLP systems
- Understand and implement research papers
- Contribute to the NLP community
- Continue learning as the field evolves

**The field moves fast—stay curious and keep learning!**

---

**Estimated Time**: 20-25 hours (including final project)

**Final Checkpoint**:
- [ ] Fine-tuned multilingual model
- [ ] Evaluated cross-lingual transfer
- [ ] Completed final project
- [ ] Reviewed course concepts
- [ ] Identified next learning goals
