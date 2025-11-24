# Week 11: Debugging & Interpretation

## Overview

This week explores how to understand what neural NLP models have learned and how they make decisions. We'll cover probing, attention analysis, mechanistic interpretability, and model editing techniques.

## Learning Objectives

By the end of this week, you will be able to:
- Design and implement probing classifiers
- Analyze attention patterns meaningfully
- Understand mechanistic interpretability concepts
- Apply model editing techniques (ROME)
- Debug model failures systematically
- Interpret what different layers/components learn

## Core Topics

### 1. Why Interpretability?

**Motivations**:
- **Safety**: Understand failure modes
- **Trust**: Explain decisions to users
- **Science**: Understand what models learn
- **Improvement**: Identify and fix issues

**Types of Interpretability**:
- **Post-hoc**: Analyze trained model
- **Intrinsic**: Model designed to be interpretable
- **Local**: Explain specific predictions
- **Global**: Understand overall model behavior

### 2. Probing Classifiers

**Key Idea**: Test if representations encode specific properties

**Method**:
```python
# 1. Extract hidden states from frozen model
# 2. Train simple classifier on those states
# 3. If classifier succeeds, info is present

class ProbingClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        return self.classifier(hidden_states)

# Usage
model.eval()
with torch.no_grad():
    hidden_states = model(input_ids, output_hidden_states=True).hidden_states[layer]

probe = ProbingClassifier(768, num_pos_tags)
# Train probe on hidden_states, labels
```

**Edge Probing Tasks** (Tenney et al., 2019):
- Part-of-speech tagging
- Dependency parsing
- Named entity recognition
- Semantic role labeling
- Coreference resolution

**Findings**:
- Lower layers: morphology, POS
- Middle layers: syntax
- Upper layers: semantics

**Control Tasks**:
- Random labels: probe should fail
- Validates that success is due to learned features

### 3. Attention Analysis

**Caution**: Attention ≠ Explanation

**What Attention Shows**:
- Where model "looks"
- Information flow patterns
- NOT necessarily causal importance

**Visualization**:
```python
from bertviz import head_view, model_view

# Get attention weights
outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)

# Visualize
head_view(attentions, tokens)
```

**Patterns to Look For**:
- Vertical stripes: attention to specific positions (CLS, SEP)
- Diagonal: sequential attention
- Specific heads attending to specific relations

**Better Alternatives**:
- Attention rollout (aggregate across layers)
- Gradient-based attribution
- Integrated gradients

### 4. Mechanistic Interpretability

**Goal**: Reverse-engineer model algorithms

**Key Concepts**:

**Circuits** (Elhage et al., 2021)
- Subgraphs that implement specific behaviors
- Can be studied in isolation
- Example: "Induction heads" for in-context learning

**Induction Heads** (Olsson et al., 2022)
Pattern: [A][B]...[A] → predicts [B]
```
"The cat sat on the mat. The cat" → likely to predict "sat"
```

**Composition**:
- Layers build on each other
- Early layers: patterns
- Later layers: compositions of patterns

**Activation Patching**:
```python
def activation_patch(model, clean_input, corrupt_input, layer, position):
    # Run corrupt forward pass, save activations
    corrupt_cache = {}
    def save_hook(module, input, output):
        corrupt_cache['activation'] = output

    # Run clean pass, patch in corrupt activation
    def patch_hook(module, input, output):
        output[:, position] = corrupt_cache['activation'][:, position]
        return output

    # Measure effect on output
```

### 5. Model Editing (ROME)

**Goal**: Change specific facts in model

**Example**:
```
Before: "The Eiffel Tower is located in" → "Paris"
After:  "The Eiffel Tower is located in" → "Rome"
```

**ROME** (Rank-One Model Editing) - Meng et al., 2022

**Key Insight**: Facts stored in MLP layers as key-value pairs

```python
# Simplified ROME update
# Find the layer/position where fact is stored
# Compute update to MLP weights

def rome_edit(model, subject, relation, new_object):
    # 1. Compute key vector k for subject
    k = get_subject_representation(model, subject)

    # 2. Compute value vector v for new fact
    v = compute_new_value(model, subject, relation, new_object)

    # 3. Update MLP weights
    # W_new = W + (v - W @ k) @ k^T / (k^T @ k)
    pass
```

**MEMIT**: Extend to multiple edits

**Limitations**:
- Can cause side effects
- May not generalize to rephrasings
- Doesn't scale well

### 6. Debugging Strategies

**When Model Fails**:

1. **Analyze Errors**
```python
# Collect failure cases
errors = []
for example in test_set:
    pred = model.predict(example)
    if pred != example.label:
        errors.append({
            'input': example.input,
            'predicted': pred,
            'expected': example.label
        })

# Categorize errors
```

2. **Adversarial Probing**
```python
# Test specific capabilities
def test_negation():
    pos = "I love this movie"
    neg = "I don't love this movie"
    assert model.predict(pos) != model.predict(neg)
```

3. **Ablation Studies**
```python
# Remove components and measure impact
def ablate_head(model, layer, head):
    model.transformer.h[layer].attn.c_attn.weight[:, head*64:(head+1)*64] = 0
    return evaluate(model)
```

4. **Minimal Pairs**
```python
# Test specific changes
pairs = [
    ("The cat chased the dog", "The dog chased the cat"),  # word order
    ("She went to the store", "He went to the store"),     # gender
]
```

### 7. Attribution Methods

**Gradient-based Attribution**
```python
def gradient_attribution(model, input_ids, target_class):
    embeddings = model.embeddings(input_ids)
    embeddings.requires_grad = True

    output = model(inputs_embeds=embeddings)
    output[0, target_class].backward()

    attribution = (embeddings.grad * embeddings).sum(-1)
    return attribution
```

**Integrated Gradients**
- Accumulate gradients along path from baseline
- More robust than simple gradients

**SHAP Values**
- Game-theoretic approach
- Measures feature contribution
- Expensive but principled

### 8. Representation Analysis

**CKA (Centered Kernel Alignment)**
- Compare representations across models/layers
- Are different models learning similar things?

**SVCCA (Singular Vector CCA)**
- Compare activation spaces
- Find shared structure

**Probing Classifier Comparison**
- Which layer encodes which property?
- How does this differ across models?

## Key References

### Required Reading

1. **Tenney et al. (2019) - "BERT Rediscovers the Classical NLP Pipeline"**
   - Edge probing analysis
   - [Paper](https://arxiv.org/abs/1905.05950)

2. **Elhage et al. (2021) - "A Mathematical Framework for Transformer Circuits"**
   - Mechanistic interpretability
   - [Paper](https://transformer-circuits.pub/2021/framework/index.html)

3. **Meng et al. (2022) - "Locating and Editing Factual Associations in GPT"**
   - ROME paper
   - [Paper](https://arxiv.org/abs/2202.05262)

### Recommended Reading

4. **Olsson et al. (2022) - "In-context Learning and Induction Heads"**
   - Induction heads discovery
   - [Paper](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

5. **Hernandez et al. (2023) - "Inspecting and Editing Knowledge Representations in Language Models"**
   - Knowledge neurons
   - [Paper](https://arxiv.org/abs/2304.00740)

6. **Jain & Wallace (2019) - "Attention is not Explanation"**
   - Critique of attention as explanation
   - [Paper](https://arxiv.org/abs/1902.10186)

## Practical Exercise

### Exercise 1: Edge Probing

**Objective**: Probe BERT for linguistic properties

```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')

# Extract representations for each layer
def get_layer_representations(text, layer):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer]

# Train probing classifiers for:
# 1. POS tagging
# 2. Named entity recognition
# 3. Dependency relation

# Compare accuracy across layers
```

**Analysis**:
- Plot accuracy vs layer
- Which layers encode which properties?
- Compare to random baseline

### Exercise 2: Model Editing with ROME

**Objective**: Edit a fact in GPT-2

```python
# Before: "The capital of France is" → "Paris"
# After: "The capital of France is" → "London"

# Use rome library or implement simplified version
from rome import ROMEHyperParams, apply_rome_to_model

# Apply edit
edited_model = apply_rome_to_model(
    model,
    tok,
    requests=[{
        "prompt": "The capital of France is",
        "subject": "France",
        "target_new": "London"
    }]
)

# Test
print(generate(edited_model, "The capital of France is"))  # Should say London
print(generate(edited_model, "Paris is located in"))       # Check side effects
```

### Exercise 3: Attention Analysis

**Objective**: Analyze attention patterns in BERT

```python
from bertviz import head_view
import seaborn as sns

# Get attention weights
outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions

# Analyze specific phenomena:
# 1. CLS token attention patterns
# 2. Positional patterns
# 3. Subject-verb agreement

def analyze_attention(text, target_token_idx):
    # Which tokens attend most to target?
    pass
```

### Exercise Files
- [exercises/probing_classifier.py](./exercises/probing_classifier.py)
- [exercises/rome_editing.py](./exercises/rome_editing.py)
- [exercises/attention_analysis.py](./exercises/attention_analysis.py)

## Study Questions

1. What does a probing classifier actually prove?
2. Why is attention not sufficient for explanation?
3. How do induction heads enable in-context learning?
4. What are the limitations of model editing?
5. How would you diagnose a model that fails on negation?

## Additional Resources

### Libraries
- [BertViz](https://github.com/jessevig/bertviz) - Attention visualization
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Mechanistic interpretability
- [rome](https://github.com/kmeng01/rome) - Model editing

### Tutorials
- [Anthropic's Transformer Circuits Thread](https://transformer-circuits.pub/)
- [Neel Nanda's Mechanistic Interpretability](https://www.neelnanda.io/mechanistic-interpretability)

### Talks
- Chris Olah's talks on neural network interpretability
- Anthropic research presentations

## Next Week Preview

Week 12 covers advanced LLMs and agents:
- Modern LLMs (LLaMA, GPT-4, Claude)
- Long context handling
- Tool use and agents

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 12, ensure you can:
- [ ] Design and train probing classifiers
- [ ] Analyze attention patterns critically
- [ ] Explain induction heads concept
- [ ] Apply ROME for model editing
- [ ] Systematically debug model failures
