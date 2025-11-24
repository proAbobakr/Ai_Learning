# Week 5: Transformers & Attention Mechanisms

## Overview

This week covers the Transformer architecture, which has revolutionized NLP since 2017. We'll understand attention mechanisms from their origins in sequence-to-sequence models through to modern self-attention and multi-head attention.

## Learning Objectives

By the end of this week, you will be able to:
- Explain different types of attention mechanisms
- Implement scaled dot-product attention from scratch
- Understand multi-head attention and its benefits
- Describe positional encoding methods
- Distinguish encoder-decoder vs decoder-only architectures
- Build a Transformer encoder for text classification

## Core Topics

### 1. Attention Mechanisms: Origins

**Sequence-to-Sequence Problem**
- RNN encoder compresses entire input into fixed vector
- Information bottleneck for long sequences
- Attention allows decoder to "look back" at encoder states

**Bahdanau Attention (2015)**
```python
# Score function: additive/concat attention
score(h_decoder, h_encoder) = v^T × tanh(W_1 × h_decoder + W_2 × h_encoder)

# Attention weights
α_t = softmax(scores)

# Context vector
c_t = Σ α_t,i × h_encoder_i
```

**Luong Attention (2015)**
```python
# Score functions
dot:     score = h_decoder^T × h_encoder
general: score = h_decoder^T × W × h_encoder
concat:  score = v^T × tanh(W × [h_decoder; h_encoder])
```

### 2. Self-Attention

**Key Insight**: Attention within a single sequence

```
"The animal didn't cross the street because it was too tired"
                                           ↑
                                    What does "it" refer to?
```

Self-attention lets each word attend to all other words.

**Scaled Dot-Product Attention**
```python
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

Where:
- **Q** (Query): what am I looking for?
- **K** (Key): what do I contain?
- **V** (Value): what do I output?
- **√d_k**: scaling factor prevents softmax saturation

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### 3. Multi-Head Attention

**Motivation**: Different heads can attend to different aspects

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O

where head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
```

**Benefits**:
- Attend to different positions
- Capture different relationship types
- Increase model capacity

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(attn_output)
```

### 4. Positional Encoding

**Problem**: Self-attention is permutation-invariant
**Solution**: Inject position information

**Sinusoidal Encoding (Original Transformer)**
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
- Unique encoding for each position
- Allows model to learn relative positions
- Can extrapolate to longer sequences

**Learned Positional Embeddings**
- Train position embeddings like word embeddings
- Used in BERT, GPT
- Limited to max sequence length seen in training

### 5. Transformer Architecture

**Encoder Block**
```
Input → [Multi-Head Self-Attention] → Add & Norm
                                          ↓
      [Feed-Forward Network] → Add & Norm → Output

Feed-Forward: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

**Decoder Block** (for generation)
```
Input → [Masked Self-Attention] → Add & Norm
                                      ↓
        [Cross-Attention to Encoder] → Add & Norm
                                           ↓
               [Feed-Forward] → Add & Norm → Output
```

**Key Differences**:
- Decoder uses causal masking (can't see future)
- Decoder has cross-attention to encoder outputs

### 6. Encoder-Only vs Decoder-Only

**Encoder-Only** (BERT-style)
- Bidirectional attention
- Good for understanding tasks
- Classification, NER, QA (extractive)

**Decoder-Only** (GPT-style)
- Causal/autoregressive attention
- Good for generation
- Text completion, dialogue

**Encoder-Decoder** (T5, BART)
- Full architecture
- Good for seq2seq tasks
- Translation, summarization

## Key References

### Required Reading

1. **Vaswani et al. (2017) - "Attention Is All You Need"**
   - Original Transformer paper
   - [Paper](https://arxiv.org/abs/1706.03762)

2. **Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate"**
   - Introduces attention for NMT
   - [Paper](https://arxiv.org/abs/1409.0473)

3. **Jay Alammar - "The Illustrated Transformer"**
   - Excellent visual explanation
   - [Blog](https://jalammar.github.io/illustrated-transformer/)

### Recommended Reading

4. **Luong et al. (2015) - "Effective Approaches to Attention-based Neural Machine Translation"**
   - Different attention variants
   - [Paper](https://arxiv.org/abs/1508.04025)

5. **Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"**
   - Encoder-only architecture
   - [Paper](https://arxiv.org/abs/1810.04805)

6. **Radford et al. (2018) - "Improving Language Understanding by Generative Pre-Training"**
   - GPT-1 paper, decoder-only
   - [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## Practical Exercise

### Exercise 1: Attention from Scratch

**Part A: Implement Scaled Dot-Product Attention**
```python
def attention(Q, K, V, mask=None):
    # Your implementation
    pass
```

**Part B: Implement Multi-Head Attention**
```python
class MultiHeadAttention(nn.Module):
    # Your implementation
    pass
```

**Part C: Visualize Attention Weights**
- Plot attention heatmaps
- Analyze what the model attends to

### Exercise 2: Transformer Encoder for Classification

**Objective**: Build a Transformer encoder and compare with LSTM

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
```

**Tasks**:
1. Train on IMDB or AG News
2. Compare accuracy and training time with LSTM
3. Analyze attention patterns

### Exercise 3: Positional Encoding

**Part A: Implement Sinusoidal Encoding**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # Your implementation
        pass
```

**Part B: Visualize Encodings**
- Plot the sinusoidal patterns
- Show how different positions are encoded

### Exercise Files
- [exercises/attention.py](./exercises/attention.py)
- [exercises/transformer_classifier.py](./exercises/transformer_classifier.py)
- [exercises/positional_encoding.py](./exercises/positional_encoding.py)
- [exercises/attention_visualization.py](./exercises/attention_visualization.py)

## Study Questions

1. Why scale by √d_k in attention?
2. How does multi-head attention increase expressivity?
3. Why use sinusoidal positional encodings instead of learned?
4. What's the computational complexity of self-attention vs RNNs?
5. When would you choose encoder-only vs decoder-only architectures?

## Additional Resources

### Visualizations
- [Attention Visualization Tool](https://github.com/jessevig/bertviz)
- [Tensor2Tensor Transformer](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)

### Tutorials
- [Harvard NLP "The Annotated Transformer"](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

### Code
- [Hugging Face Transformers source](https://github.com/huggingface/transformers)
- [minGPT](https://github.com/karpathy/minGPT)

## Next Week Preview

Week 6 covers text generation algorithms and prompting:
- Decoding strategies (greedy, beam, sampling)
- Temperature and top-k/top-p sampling
- Introduction to prompt engineering

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 6, ensure you can:
- [ ] Implement scaled dot-product attention
- [ ] Explain multi-head attention mechanism
- [ ] Build a Transformer encoder classifier
- [ ] Compare Transformer vs LSTM performance
- [ ] Visualize and interpret attention weights
