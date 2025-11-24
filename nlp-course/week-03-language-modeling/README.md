# Week 3: Language Modeling

## Overview

Language modeling is the task of predicting the probability of a sequence of words. It's fundamental to modern NLP, underpinning everything from speech recognition to text generation. This week covers both classical n-gram approaches and neural language models.

## Learning Objectives

By the end of this week, you will be able to:
- Explain the mathematical foundation of language modeling
- Implement n-gram models with various smoothing techniques
- Understand the limitations of count-based approaches
- Build a neural feed-forward language model
- Calculate and interpret perplexity
- Compare different language modeling approaches

## Core Topics

### 1. Foundations of Language Modeling

**The Language Modeling Task**
```
P(w_1, w_2, ..., w_n) = P(w_1) × P(w_2|w_1) × P(w_3|w_1,w_2) × ...
```

Using the chain rule of probability, we decompose joint probability into conditional probabilities.

**Markov Assumption**
- Full history is intractable
- Approximate by limiting context
- N-gram: condition on previous n-1 words

```
Bigram:  P(w_i | w_1...w_{i-1}) ≈ P(w_i | w_{i-1})
Trigram: P(w_i | w_1...w_{i-1}) ≈ P(w_i | w_{i-2}, w_{i-1})
```

### 2. N-gram Language Models

**Maximum Likelihood Estimation**
```
P_MLE(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
```

**The Zero Probability Problem**
- Unseen n-grams get P = 0
- Entire sequence probability becomes 0
- Need smoothing techniques

### 3. Smoothing Techniques

**Add-k Smoothing (Laplace)**
```
P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k×V)
```
- Simple but not optimal
- k = 1 is Laplace smoothing
- k < 1 often works better

**Good-Turing Smoothing**
- Use count of counts
- Reallocate probability mass from seen to unseen
- N_c = number of n-grams seen c times

**Kneser-Ney Smoothing**
- State-of-the-art for n-grams
- Absolute discounting + continuation probability
- Accounts for word versatility

```
P_KN(w_i | w_{i-1}) = max(count(w_{i-1},w_i) - d, 0) / count(w_{i-1})
                     + λ(w_{i-1}) × P_continuation(w_i)
```

**Interpolation & Backoff**
- Combine different order n-grams
- Backoff: use lower order when higher order unseen
- Interpolation: weighted combination always

### 4. Evaluation: Perplexity

**Definition**
```
Perplexity(W) = P(w_1, w_2, ..., w_N)^{-1/N}
              = exp(- (1/N) Σ log P(w_i | w_{<i}))
```

**Interpretation**
- Average branching factor
- Lower is better
- Typical values: 50-200 for word-level models

**Cross-Entropy Relationship**
```
Perplexity = 2^{cross-entropy}
```

### 5. Neural Language Models

**Feed-Forward Neural LM (Bengio et al., 2003)**
```
Input: [w_{i-3}, w_{i-2}, w_{i-1}] (fixed context window)
       ↓
Embedding lookup
       ↓
Concatenate embeddings
       ↓
Hidden layer (tanh)
       ↓
Output layer (softmax over vocabulary)
       ↓
P(w_i | context)
```

**Advantages over N-grams**
- Learns word similarities through embeddings
- Shares parameters across positions
- Better generalization to unseen contexts

**Challenges**
- Fixed context window
- Computational cost of softmax over vocabulary
- Will address with RNNs (Week 4) and Transformers (Week 5)

### 6. Practical Considerations

**Vocabulary**
- Unknown word token `<UNK>`
- Sentence boundaries `<s>`, `</s>`
- Subword units can eliminate OOV

**Out-of-Vocabulary Handling**
- Replace rare words with `<UNK>`
- Use subword tokenization
- Character-level models

## Key References

### Required Reading

1. **Jurafsky & Martin - Chapter 3: N-gram Language Models**
   - Comprehensive coverage of n-gram LMs
   - Smoothing techniques explained
   - [Online chapter](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

2. **Goodman (1998) - "A Bit of Progress in Language Modeling"**
   - Classic survey of smoothing techniques
   - Experimental comparisons

3. **Bengio et al. (2003) - "A Neural Probabilistic Language Model"**
   - First neural language model paper
   - Foundation for modern approaches
   - [Paper](https://www.jmlr.org/papers/v3/bengio03a.html)

### Recommended Reading

4. **Chen & Goodman (1999) - "An Empirical Study of Smoothing Techniques"**
   - Detailed comparison of smoothing methods
   - Practical recommendations

5. **Heafield (2011) - "KenLM: Faster and Smaller Language Model Queries"**
   - Efficient n-gram LM toolkit
   - [Paper](https://aclanthology.org/W11-2123/)

### Tools

- **KenLM**: Fast n-gram language modeling toolkit
- **SRILM**: Stanford Research Institute Language Modeling toolkit

## Practical Exercise

### Exercise 1: N-gram Language Model from Scratch

**Part A: Basic Implementation**
```python
class NGramLM:
    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()

    def train(self, corpus):
        # Count n-grams and contexts
        pass

    def probability(self, word, context):
        # Return P(word | context)
        pass

    def perplexity(self, test_corpus):
        # Calculate perplexity on test set
        pass
```

**Part B: Add Smoothing**
- Implement add-k smoothing
- Implement interpolation with lower-order models
- Compare perplexity with different methods

**Part C: Text Generation**
```python
def generate(self, context, num_words):
    # Sample from the model
    pass
```

### Exercise 2: Feed-Forward Neural LM

**Objective**: Implement Bengio's neural language model in PyTorch

**Architecture**:
```python
class FFNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size):
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(context_size * embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        # context: [batch, context_size]
        embeds = self.embeddings(context)  # [batch, context, embed]
        embeds = embeds.view(embeds.size(0), -1)  # flatten
        hidden = torch.tanh(self.hidden(embeds))
        logits = self.output(hidden)
        return logits
```

**Tasks**:
1. Train on a text corpus (PTB, WikiText)
2. Compute perplexity on validation/test
3. Generate text samples
4. Compare with n-gram baseline

### Exercise 3: Perplexity Comparison

Create a table comparing:
| Model | Train PPL | Valid PPL | Test PPL |
|-------|-----------|-----------|----------|
| Bigram + Laplace | | | |
| Trigram + KN | | | |
| FFNN LM | | | |

### Exercise Files
- [exercises/ngram_lm.py](./exercises/ngram_lm.py)
- [exercises/neural_lm.py](./exercises/neural_lm.py)
- [exercises/perplexity_comparison.py](./exercises/perplexity_comparison.py)

## Study Questions

1. Why does increasing n in n-gram models help and hurt?
2. How does Kneser-Ney smoothing account for word versatility?
3. What's the relationship between perplexity and bits per word?
4. Why do neural LMs generalize better to unseen n-grams?
5. What are the computational bottlenecks in neural LMs?

## Additional Resources

### Tutorials
- [Stanford CS224N: Language Models](https://www.youtube.com/watch?v=OQQ-W_63UgQ)
- [N-gram tutorial with Python](https://www.nltk.org/book/ch02.html)

### Datasets
- Penn Treebank (PTB)
- WikiText-2 and WikiText-103
- 1 Billion Word Benchmark

### Tools
- `nltk.lm` - NLTK language modeling module
- KenLM - Command-line n-gram toolkit

## Next Week Preview

Week 4 introduces Recurrent Neural Networks (RNNs), which can model sequences of arbitrary length:
- Vanilla RNNs and the vanishing gradient problem
- LSTM and GRU architectures
- Bidirectional models

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 4, ensure you can:
- [ ] Implement an n-gram LM with smoothing
- [ ] Calculate perplexity manually and programmatically
- [ ] Explain why smoothing is necessary
- [ ] Train a feed-forward neural LM
- [ ] Compare perplexity across different models
