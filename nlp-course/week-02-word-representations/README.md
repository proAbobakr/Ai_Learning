# Week 2: Word Representations & Text Classification

## Overview

This week explores how to represent words and text as numerical vectors that machine learning models can process. We'll cover both traditional approaches (bag-of-words) and neural methods (word embeddings), with a focus on practical text classification.

## Learning Objectives

By the end of this week, you will be able to:
- Implement bag-of-words and TF-IDF representations
- Understand and apply subword tokenization (BPE, SentencePiece)
- Explain how Word2Vec and GloVe learn word embeddings
- Visualize embeddings using t-SNE and PCA
- Build text classifiers using CNNs or LSTMs with learned embeddings

## Core Topics

### 1. Traditional Text Representations

**Bag-of-Words (BoW)**
- Document as unordered collection of words
- Vocabulary creation and indexing
- Count vectors and binary vectors
- Limitations: no word order, no semantics

**TF-IDF (Term Frequency-Inverse Document Frequency)**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

TF(t,d) = count(t in d) / |d|
IDF(t) = log(N / df(t))
```
- Balances word frequency with document frequency
- Reduces impact of common words
- Still used in information retrieval

### 2. Subword Tokenization

**Why Subwords?**
- Handle out-of-vocabulary (OOV) words
- Reduce vocabulary size
- Capture morphological structure
- Enable cross-lingual transfer

**Byte Pair Encoding (BPE)**
```
Algorithm:
1. Start with character vocabulary
2. Count adjacent symbol pairs
3. Merge most frequent pair
4. Repeat until vocabulary size reached
```

**SentencePiece**
- Language-agnostic tokenization
- Treats input as raw byte sequence
- Unigram language model variant
- Used by many modern models (T5, LLaMA)

**WordPiece**
- Similar to BPE, used by BERT
- Merges based on likelihood increase
- Special tokens: [CLS], [SEP], [PAD]

### 3. Neural Word Embeddings

**Word2Vec (Mikolov et al., 2013)**

*Skip-gram*: Predict context from word
```
Input: "cat" → Output: ["the", "sat", "on", "mat"]
```

*CBOW*: Predict word from context
```
Input: ["the", "sat", "on", "mat"] → Output: "cat"
```

**Key Properties**
- Dense, low-dimensional vectors (50-300 dims)
- Semantic similarity through vector distance
- Analogical reasoning: king - man + woman ≈ queen

**GloVe (Pennington et al., 2014)**
- Global Vectors for Word Representation
- Combines count-based and predictive methods
- Factorizes co-occurrence matrix
- Objective: `w_i · w_j + b_i + b_j = log(X_ij)`

### 4. Embedding Visualization

**Principal Component Analysis (PCA)**
- Linear dimensionality reduction
- Preserves global structure
- Fast computation
- Good for initial exploration

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Non-linear dimensionality reduction
- Preserves local structure
- Reveals clusters
- Slower, hyperparameter sensitive

**UMAP**
- Faster alternative to t-SNE
- Better preservation of global structure
- Often preferred for large datasets

### 5. Text Classification Architectures

**CNN for Text (Kim, 2014)**
```
Embedding → Conv1D filters → Max pooling → Dense → Output
            (multiple sizes)
```
- Captures local n-gram patterns
- Parallel filter application
- Position-invariant features

**LSTM for Text**
```
Embedding → LSTM → (Hidden state) → Dense → Output
```
- Captures sequential dependencies
- Better for longer documents
- Can use bidirectional variant

## Key References

### Required Reading

1. **Sennrich et al. (2015) - "Neural Machine Translation of Rare Words with Subword Units"**
   - Introduces BPE for NMT
   - [Paper](https://aclanthology.org/P16-1162/)

2. **Kudo (2018) - "Subword Regularization"**
   - SentencePiece algorithm
   - [Paper](https://aclanthology.org/P18-1007/)

3. **Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"**
   - Original Word2Vec paper
   - [Paper](https://arxiv.org/abs/1301.3781)

4. **Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"**
   - GloVe embeddings
   - [Paper](https://aclanthology.org/D14-1162/)

### Recommended Reading

5. **Bengio et al. (2003) - "A Neural Probabilistic Language Model"**
   - Foundational neural LM paper
   - First learned word embeddings
   - [Paper](https://www.jmlr.org/papers/v3/bengio03a.html)

6. **Kim (2014) - "Convolutional Neural Networks for Sentence Classification"**
   - TextCNN architecture
   - [Paper](https://aclanthology.org/D14-1181/)

7. **Karpathy's Blog - "Hacker's Guide to Neural Networks"**
   - Intuitive backpropagation explanation
   - [Blog](http://karpathy.github.io/neuralnets/)

## Practical Exercise

### Exercise 1: Subword Tokenization

**Part A: Train a SentencePiece Model**
```python
import sentencepiece as spm

# Train on your corpus
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_tokenizer',
    vocab_size=8000,
    model_type='bpe'  # or 'unigram'
)
```

**Part B: Compare Tokenizations**
- Compare character, word, BPE, and unigram tokenization
- Analyze how OOV words are handled
- Measure vocabulary coverage on test set

### Exercise 2: Text Classification with Embeddings

**Objective**: Build a CNN text classifier using SentencePiece tokenization

**Steps**:
1. Preprocess dataset with SentencePiece
2. Load pre-trained embeddings or train from scratch
3. Implement CNN classifier (multiple filter sizes)
4. Compare with bag-of-words baseline
5. Visualize learned embeddings

**Dataset Options**:
- IMDB sentiment
- AG News topic classification
- SST-2 (Stanford Sentiment Treebank)

**Evaluation**:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Error analysis

### Exercise 3: Embedding Exploration

**Part A: Analogy Tasks**
```python
# Test: king - man + woman = ?
# Test: Paris - France + Germany = ?
```

**Part B: Visualization**
- Create t-SNE plots of word categories
- Compare semantic clusters
- Identify embedding biases

### Exercise Files
- [exercises/subword_tokenization.py](./exercises/subword_tokenization.py)
- [exercises/cnn_classifier.py](./exercises/cnn_classifier.py)
- [exercises/embedding_visualization.py](./exercises/embedding_visualization.py)

## Study Questions

1. Why do subword methods handle morphologically rich languages better?
2. What are the trade-offs between Word2Vec and GloVe?
3. How does the choice of vocabulary size affect model performance?
4. Why might CNNs work well for short texts but struggle with long documents?
5. What biases might be captured in word embeddings trained on web text?

## Additional Resources

### Visualizations
- [Embedding Projector](https://projector.tensorflow.org/) - Google's interactive tool
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Jay Alammar

### Pre-trained Embeddings
- [GloVe vectors](https://nlp.stanford.edu/projects/glove/)
- [FastText vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)

### Tutorials
- [SentencePiece documentation](https://github.com/google/sentencepiece)
- [PyTorch text classification tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

## Next Week Preview

Week 3 focuses on Language Modeling - predicting the next word in a sequence. You'll learn about:
- N-gram language models and smoothing
- Neural language models
- Perplexity as an evaluation metric

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 3, ensure you can:
- [ ] Train a SentencePiece tokenizer on custom data
- [ ] Explain the difference between BPE and unigram models
- [ ] Load and use pre-trained word embeddings
- [ ] Build a working CNN text classifier
- [ ] Create embedding visualizations with t-SNE
