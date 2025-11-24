# Week 1: Introduction & NLP Fundamentals

## Overview

This week introduces the field of Natural Language Processing, tracing its evolution from rule-based systems to modern neural approaches. You'll gain perspective on where NLP has been and where it's heading, while getting hands-on with your first text classification systems.

## Learning Objectives

By the end of this week, you will be able to:
- Describe the historical evolution of NLP approaches
- Identify common NLP tasks and their applications
- Understand the shift from symbolic to statistical to neural methods
- Implement a simple rule-based text classifier
- Compare rule-based approaches with statistical methods
- Recognize social and ethical considerations in NLP

## Core Topics

### 1. History of NLP

**Rule-Based Era (1950s-1990s)**
- Symbolic AI and expert systems
- Hand-crafted rules and grammars
- ELIZA, SHRDLU, and early chatbots
- Limitations: brittleness, lack of coverage

**Statistical Era (1990s-2010s)**
- Corpus-based approaches
- Hidden Markov Models (HMMs)
- Maximum Entropy models
- Conditional Random Fields (CRFs)
- Key shift: learning patterns from data

**Neural Era (2010s-Present)**
- Word embeddings (Word2Vec, 2013)
- Recurrent Neural Networks
- Attention mechanisms (2015)
- Transformers (2017)
- Large Language Models (2018+)

### 2. Common NLP Tasks

**Understanding Tasks**
- **Text Classification**: Sentiment analysis, spam detection, topic categorization
- **Named Entity Recognition (NER)**: Identifying people, places, organizations
- **Part-of-Speech Tagging**: Grammatical role assignment
- **Parsing**: Syntactic structure analysis
- **Coreference Resolution**: Linking mentions to entities

**Generation Tasks**
- **Machine Translation**: Converting between languages
- **Summarization**: Condensing documents
- **Question Answering**: Responding to queries
- **Dialogue Systems**: Conversational AI
- **Text Generation**: Creative and functional writing

### 3. The Modern NLP Pipeline

```
Raw Text → Preprocessing → Tokenization → Representation → Model → Output
           (cleaning)      (splitting)    (embeddings)    (neural)  (task)
```

### 4. Social Considerations in NLP

- Bias in language models and training data
- Representation and fairness across demographics
- Privacy concerns with language data
- Environmental impact of large model training
- Misuse potential and safeguards

## Key References

### Required Reading

1. **CMU 11-711 Lecture 1 - Introduction**
   - Course overview and NLP landscape
   - [Course website](http://phontron.com/class/anlp2024/)

2. **Sap et al. (2017) - "Connotation Frames of Power and Agency in Modern Films"**
   - Examines social bias and agency in language
   - Demonstrates how NLP can reveal societal patterns
   - [Paper link](https://aclanthology.org/D17-1247/)

3. **Jurafsky & Martin - Chapter 1-2**
   - Speech and Language Processing (3rd ed.)
   - Introduction to NLP and regular expressions
   - [Online textbook](https://web.stanford.edu/~jurafsky/slp3/)

### Recommended Reading

4. **Manning & Schütze - "Foundations of Statistical Natural Language Processing"**
   - Chapters 1-2 for historical context
   - Classic textbook on pre-neural NLP

5. **Eisenstein - "Introduction to Natural Language Processing"**
   - Chapter 1: Introduction
   - Modern perspective on the field

## Practical Exercise

### Exercise 1: Rule-Based vs. Statistical Classification

**Objective**: Implement both a rule-based and a logistic regression classifier for sentiment analysis, then compare their performance.

**Dataset**: Use a simple sentiment dataset (e.g., movie reviews, product reviews)

**Part A: Rule-Based Classifier**
```python
# Implement a classifier using:
# - Positive/negative word lists
# - Negation handling
# - Intensity modifiers
```

**Part B: Logistic Regression Classifier**
```python
# Implement using:
# - Bag-of-words features
# - TF-IDF weighting
# - scikit-learn LogisticRegression
```

**Part C: Comparison**
- Compare accuracy, precision, recall, F1
- Analyze failure cases for each approach
- Discuss strengths and weaknesses

**Deliverables**:
1. Working code for both classifiers
2. Evaluation metrics comparison table
3. Written analysis (1-2 paragraphs) of observations

### Exercise Files
- [exercises/rule_based_classifier.py](./exercises/rule_based_classifier.py)
- [exercises/statistical_classifier.py](./exercises/statistical_classifier.py)
- [exercises/comparison_analysis.md](./exercises/comparison_analysis.md)

## Study Questions

1. Why did the field shift from rule-based to statistical approaches?
2. What are the trade-offs between interpretability and performance in NLP?
3. How do modern neural approaches combine aspects of both paradigms?
4. What ethical considerations should guide NLP research and deployment?
5. How has the availability of data and compute influenced NLP progress?

## Additional Resources

### Videos
- Stanford CS224N Lecture 1: Introduction to NLP
- 3Blue1Brown: Neural Networks series (background)

### Blog Posts
- "A Visual Survey of Data Augmentation in NLP" - Amit Chaudhary
- "The Illustrated Word2Vec" - Jay Alammar (preview for Week 2)

### Tools to Explore
- NLTK (Natural Language Toolkit)
- spaCy for preprocessing
- Hugging Face datasets library

## Next Week Preview

In Week 2, we'll dive into word representations, exploring how to convert text into numerical vectors that capture meaning. You'll learn about:
- Bag-of-words and TF-IDF
- Subword tokenization (BPE, SentencePiece)
- Word embeddings (Word2Vec, GloVe)

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 2, ensure you can:
- [ ] Explain the three eras of NLP development
- [ ] List 5+ common NLP tasks and their applications
- [ ] Implement a simple rule-based text classifier
- [ ] Train and evaluate a logistic regression classifier
- [ ] Discuss at least 2 ethical considerations in NLP
