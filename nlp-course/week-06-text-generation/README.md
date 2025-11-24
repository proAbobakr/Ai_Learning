# Week 6: Text Generation Algorithms & Prompting Basics

## Overview

This week explores how to generate text from language models, covering various decoding strategies from simple greedy search to sophisticated sampling methods. We'll also introduce prompt engineering fundamentals.

## Learning Objectives

By the end of this week, you will be able to:
- Implement greedy and beam search decoding
- Understand and apply temperature scaling
- Use top-k and nucleus (top-p) sampling
- Recognize the trade-offs between different decoding methods
- Write effective prompts for language models
- Experiment with prompt variations to control output

## Core Topics

### 1. The Generation Problem

**Autoregressive Generation**
```
P(w_1, w_2, ..., w_n) = Π P(w_t | w_1, ..., w_{t-1})
```

At each step, predict next token from vocabulary.

**The Challenge**
- Vocabulary size V (e.g., 50,000 tokens)
- Sequence length N
- Total possible sequences: V^N (intractable)
- Need efficient search strategies

### 2. Greedy Decoding

**Algorithm**
```python
def greedy_decode(model, prompt, max_length):
    tokens = prompt
    for _ in range(max_length):
        logits = model(tokens)
        next_token = logits[-1].argmax()
        tokens = torch.cat([tokens, next_token])
        if next_token == EOS:
            break
    return tokens
```

**Properties**
- Fast: O(N) model calls
- Deterministic
- Often produces repetitive, generic text
- Doesn't explore alternatives

### 3. Beam Search

**Idea**: Track top-k partial hypotheses

```python
def beam_search(model, prompt, beam_width, max_length):
    beams = [(prompt, 0.0)]  # (sequence, log_prob)

    for _ in range(max_length):
        candidates = []
        for seq, score in beams:
            logits = model(seq)
            log_probs = F.log_softmax(logits[-1], dim=-1)
            top_k = log_probs.topk(beam_width)

            for prob, token in zip(top_k.values, top_k.indices):
                new_seq = torch.cat([seq, token])
                new_score = score + prob
                candidates.append((new_seq, new_score))

        # Keep top beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return beams[0][0]
```

**Properties**
- Better than greedy for translation, summarization
- Still deterministic
- Can produce degenerate text for open-ended generation
- Length normalization often needed

### 4. Temperature Scaling

**Concept**: Control randomness in distribution

```python
logits = model(tokens)
scaled_logits = logits / temperature
probs = F.softmax(scaled_logits, dim=-1)
```

**Effect of Temperature**
- T = 1.0: Original distribution
- T < 1.0: Sharper distribution (more confident)
- T > 1.0: Flatter distribution (more random)
- T → 0: Approaches greedy
- T → ∞: Approaches uniform random

### 5. Top-k Sampling

**Algorithm**
```python
def top_k_sampling(logits, k):
    top_k_logits, top_k_indices = logits.topk(k)
    probs = F.softmax(top_k_logits, dim=-1)
    idx = torch.multinomial(probs, 1)
    return top_k_indices[idx]
```

**Properties**
- Restricts to k most likely tokens
- Prevents sampling very unlikely tokens
- k is fixed regardless of distribution shape
- Common values: k = 50

### 6. Nucleus (Top-p) Sampling

**Idea**: Sample from smallest set covering probability mass p

```python
def nucleus_sampling(logits, p):
    sorted_logits, sorted_indices = logits.sort(descending=True)
    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Find cutoff index
    cutoff_idx = (cumulative_probs <= p).sum()

    # Mask tokens outside nucleus
    filtered_logits = sorted_logits[:cutoff_idx + 1]
    probs = F.softmax(filtered_logits, dim=-1)

    idx = torch.multinomial(probs, 1)
    return sorted_indices[idx]
```

**Properties**
- Adapts to distribution shape
- Sharper distributions → fewer tokens sampled from
- Flatter distributions → more tokens sampled from
- Common values: p = 0.9, 0.95

### 7. Combining Strategies

**Typical Configuration**
```python
output = model.generate(
    prompt,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.1
)
```

### 8. Repetition Penalties

**Problem**: Models often repeat themselves

**Solutions**:
- **Repetition penalty**: Reduce probability of seen tokens
- **No-repeat n-gram**: Block repeated n-grams
- **Frequency penalty**: Penalize based on count
- **Presence penalty**: Penalize any repeated token

### 9. Introduction to Prompting

**What is a Prompt?**
The input text that guides model generation.

**Basic Prompt Patterns**

*Instruction*:
```
Translate English to French:
English: Hello, how are you?
French:
```

*Few-shot*:
```
Sentiment: positive
Text: I loved this movie!

Sentiment: negative
Text: Terrible waste of time.

Sentiment:
Text: It was okay, nothing special.
```

*Chain-of-thought (preview)*:
```
Q: If there are 3 cars in the parking lot and 2 more arrive, how many cars are there?
A: Let's think step by step. Initially there are 3 cars. Then 2 more arrive. 3 + 2 = 5.

Q: If there are 5 apples and you take away 2, how many do you have?
A:
```

### 10. Prompt Engineering Principles

1. **Be specific**: Clear instructions get better results
2. **Provide examples**: Show the desired format
3. **Specify format**: Request JSON, bullet points, etc.
4. **Set constraints**: Length, style, tone
5. **Iterate**: Test and refine prompts

## Key References

### Required Reading

1. **Holtzman et al. (2020) - "The Curious Case of Neural Text Degeneration"**
   - Introduces nucleus sampling
   - Analyzes failure modes of greedy/beam search
   - [Paper](https://arxiv.org/abs/1904.09751)

2. **Kool et al. (2019) - "Stochastic Beams and Where To Find Them"**
   - Stochastic beam search
   - [Paper](https://arxiv.org/abs/1903.06059)

### Recommended Reading

3. **Fan et al. (2018) - "Hierarchical Neural Story Generation"**
   - Top-k sampling introduction
   - [Paper](https://arxiv.org/abs/1805.04833)

4. **Krishna et al. (2022) - "RankGen: Improving Text Generation with Large Ranking Models"**
   - Advanced generation with ranking
   - [Paper](https://arxiv.org/abs/2205.09726)

5. **Prompt Engineering Guide**
   - Comprehensive prompt engineering resource
   - [Guide](https://www.promptingguide.ai/)

## Practical Exercise

### Exercise 1: Implement Decoding Strategies

**Part A: Greedy Decoding**
```python
def greedy_decode(model, tokenizer, prompt, max_length):
    # Your implementation
    pass
```

**Part B: Beam Search**
```python
def beam_search(model, tokenizer, prompt, beam_width, max_length):
    # Your implementation
    pass
```

**Part C: Nucleus Sampling**
```python
def nucleus_sample(model, tokenizer, prompt, top_p, temperature, max_length):
    # Your implementation
    pass
```

### Exercise 2: Compare Generation Quality

**Task**: Generate text with different strategies and compare

```python
prompt = "Once upon a time in a distant kingdom,"

strategies = [
    {"method": "greedy"},
    {"method": "beam", "beam_width": 5},
    {"method": "sample", "temperature": 0.7},
    {"method": "top_k", "k": 50},
    {"method": "nucleus", "p": 0.9},
]

for strategy in strategies:
    output = generate(prompt, **strategy)
    print(f"{strategy}: {output}")
```

**Analysis**:
- Rate each output for fluency, coherence, creativity
- Count repetitions
- Measure diversity across multiple samples

### Exercise 3: Prompt Engineering Lab

**Task**: Experiment with prompts for different outcomes

**Scenario 1: Summarization**
```python
prompts = [
    "Summarize this text: {text}",
    "Write a one-paragraph summary: {text}",
    "TL;DR: {text}",
    "Key points from this article:\n{text}\n\nKey points:",
]
```

**Scenario 2: Tone Control**
```python
base_prompt = "Write a product description for a new smartphone."

variations = [
    base_prompt,
    base_prompt + " Use an enthusiastic tone.",
    base_prompt + " Use a professional, technical tone.",
    base_prompt + " Use a casual, friendly tone.",
]
```

**Deliverable**: Report on how prompts affect output

### Exercise Files
- [exercises/decoding_strategies.py](./exercises/decoding_strategies.py)
- [exercises/generation_comparison.py](./exercises/generation_comparison.py)
- [exercises/prompt_experiments.py](./exercises/prompt_experiments.py)

## Study Questions

1. Why does beam search often produce generic text for open-ended generation?
2. How does nucleus sampling adapt to different probability distributions?
3. What's the trade-off between temperature and diversity?
4. Why might you combine top-k and nucleus sampling?
5. How do few-shot prompts enable new capabilities without fine-tuning?

## Additional Resources

### Tools
- Hugging Face `generate()` method
- OpenAI Playground (for experimenting with prompts)

### Tutorials
- [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [How to Generate Text](https://huggingface.co/blog/how-to-generate)

### Visualization
- [Text Generation Visualization](https://textgenrnn.readthedocs.io/)

## Next Week Preview

Week 7 covers instruction tuning and efficient fine-tuning:
- Full fine-tuning vs parameter-efficient methods
- LoRA, adapters, and prompt tuning
- Instruction-tuned models (FLAN, InstructGPT)

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 7, ensure you can:
- [ ] Implement greedy, beam, and sampling-based decoding
- [ ] Explain the nucleus sampling algorithm
- [ ] Use temperature to control generation diversity
- [ ] Write prompts that guide model behavior
- [ ] Analyze trade-offs between different decoding strategies
