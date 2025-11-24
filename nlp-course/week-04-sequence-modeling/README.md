# Week 4: Sequence Modeling (RNNs, LSTMs, GRUs)

## Overview

This week introduces recurrent neural networks and their variants, which can process sequences of arbitrary length. We'll understand the vanishing gradient problem and how LSTM and GRU architectures address it.

## Learning Objectives

By the end of this week, you will be able to:
- Explain how RNNs process sequential data
- Identify the vanishing and exploding gradient problems
- Implement LSTM and GRU cells from scratch
- Train recurrent language models
- Apply bidirectional RNNs for sequence labeling
- Compare RNN variants on language modeling tasks

## Core Topics

### 1. Recurrent Neural Networks

**Basic RNN Architecture**
```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

- **Hidden state** h_t captures sequence history
- **Same weights** applied at every timestep (parameter sharing)
- **Arbitrary length** sequences can be processed

**Unrolling Through Time**
```
x_1 → [RNN] → h_1 → [RNN] → h_2 → [RNN] → h_3 → ...
        ↓           ↓           ↓
       y_1         y_2         y_3
```

### 2. Backpropagation Through Time (BPTT)

**The Gradient Flow**
```
∂L/∂W = Σ_t ∂L_t/∂W

∂L_t/∂h_k = ∂L_t/∂h_t × Π_{i=k}^{t-1} ∂h_{i+1}/∂h_i
```

**Vanishing Gradient Problem**
- Gradients multiply through many timesteps
- If |∂h_{i+1}/∂h_i| < 1, gradient vanishes exponentially
- Network "forgets" long-range dependencies

**Exploding Gradient Problem**
- If |∂h_{i+1}/∂h_i| > 1, gradient explodes
- Training becomes unstable
- Solution: gradient clipping

### 3. Long Short-Term Memory (LSTM)

**Key Innovation**: Gated architecture with cell state

**LSTM Equations**
```python
# Forget gate: what to discard from cell state
f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)

# Input gate: what new information to store
i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)

# Candidate values
c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)

# Update cell state
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

# Output gate
o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)

# Hidden state
h_t = o_t ⊙ tanh(c_t)
```

**Why LSTMs Work**
- Cell state provides "highway" for gradient flow
- Gates learn what to remember/forget
- Additive updates (not multiplicative) preserve gradients

### 4. Gated Recurrent Unit (GRU)

**Simplified Gating** (fewer parameters than LSTM)

```python
# Update gate
z_t = sigmoid(W_z × [h_{t-1}, x_t])

# Reset gate
r_t = sigmoid(W_r × [h_{t-1}, x_t])

# Candidate hidden state
h̃_t = tanh(W × [r_t ⊙ h_{t-1}, x_t])

# Final hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**LSTM vs GRU**
- GRU: fewer parameters, often faster
- LSTM: more expressive, better on some tasks
- Performance often similar; try both

### 5. Bidirectional RNNs

**Motivation**: Context from both directions

```
Forward:  x_1 → h_1→ → h_2→ → h_3→
Backward: x_1 ← h_1← ← h_2← ← h_3←

Output: [h_t→; h_t←] concatenated
```

**Applications**
- Sequence labeling (NER, POS tagging)
- Sentence encoding
- Not applicable for generation (can't see future)

### 6. Practical Considerations

**Training Tips**
- Use gradient clipping (max_norm=1.0 or 5.0)
- Initialize forget gate bias to 1.0 (remember by default)
- Use dropout between layers (not within recurrence)
- Consider layer normalization

**Stacked RNNs**
```
x_t → LSTM_1 → LSTM_2 → LSTM_3 → output
```
- Deeper models capture hierarchical features
- 2-4 layers typically sufficient

## Key References

### Required Reading

1. **Elman (1990) - "Finding Structure in Time"**
   - Original RNN paper
   - Simple recurrent networks
   - [Paper](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1)

2. **Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"**
   - LSTM architecture
   - [Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)

3. **Pascanu et al. (2013) - "On the Difficulty of Training Recurrent Neural Networks"**
   - Vanishing/exploding gradients analysis
   - [Paper](https://arxiv.org/abs/1211.5063)

### Recommended Reading

4. **Cho et al. (2014) - "Learning Phrase Representations using RNN Encoder-Decoder"**
   - Introduces GRU
   - [Paper](https://arxiv.org/abs/1406.1078)

5. **Graves (2012) - "Supervised Sequence Labelling with Recurrent Neural Networks"**
   - Comprehensive book on RNNs
   - [Book](https://www.cs.toronto.edu/~graves/preprint.pdf)

6. **Karpathy (2015) - "The Unreasonable Effectiveness of Recurrent Neural Networks"**
   - Intuitive blog post with examples
   - [Blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Visualization

7. **Olah (2015) - "Understanding LSTM Networks"**
   - Excellent visual explanation
   - [Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Practical Exercise

### Exercise 1: LSTM from Scratch

**Objective**: Implement an LSTM cell without using nn.LSTM

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined weight matrix for efficiency
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.W(combined)

        # Split into 4 gates
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)     # cell candidate
        o = torch.sigmoid(o)  # output gate

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, (h, c)
```

### Exercise 2: LSTM Language Model

**Objective**: Build and train an LSTM language model

**Architecture**:
```python
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
```

**Tasks**:
1. Train on WikiText-2 or PTB
2. Implement proper batching with padding
3. Compare perplexity with feed-forward LM from Week 3
4. Visualize gradient norms during training
5. Generate text samples

### Exercise 3: Gradient Analysis

**Objective**: Empirically observe vanishing gradients

1. Train vanilla RNN and LSTM on same task
2. Log gradient norms at each layer over time
3. Plot gradient flow through sequence length
4. Compare learning curves

### Exercise 4: Bidirectional Sequence Labeling

**Objective**: POS tagging with BiLSTM

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim, hidden_dim):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
```

### Exercise Files
- [exercises/lstm_from_scratch.py](./exercises/lstm_from_scratch.py)
- [exercises/lstm_language_model.py](./exercises/lstm_language_model.py)
- [exercises/gradient_analysis.py](./exercises/gradient_analysis.py)
- [exercises/bilstm_tagger.py](./exercises/bilstm_tagger.py)

## Study Questions

1. Why does parameter sharing across timesteps help generalization?
2. How does the cell state in LSTM help with long-range dependencies?
3. When would you prefer GRU over LSTM?
4. Why can't we use bidirectional RNNs for language modeling?
5. What's the computational complexity of BPTT?

## Additional Resources

### Visualizations
- [LSTM Visualization](https://lstm.seas.harvard.edu/)
- [Recurrent Neural Network Playground](https://distill.pub/2019/memorization-in-rnns/)

### Tutorials
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
- [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)

### Datasets
- WikiText-2, WikiText-103
- Penn Treebank
- Universal Dependencies (for sequence labeling)

## Next Week Preview

Week 5 introduces Transformers, which have largely replaced RNNs in modern NLP:
- Self-attention mechanism
- Multi-head attention
- Positional encodings
- The original Transformer architecture

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 5, ensure you can:
- [ ] Implement an LSTM cell from scratch
- [ ] Explain how gates help gradient flow
- [ ] Train an LSTM language model
- [ ] Achieve lower perplexity than feed-forward LM
- [ ] Apply BiLSTM for sequence labeling
