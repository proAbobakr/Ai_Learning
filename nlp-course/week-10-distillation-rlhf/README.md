# Week 10: Distillation, Quantization & RLHF

## Overview

This week covers two major topics: model compression (making models smaller/faster) and alignment (making models helpful and harmless). We'll explore distillation, quantization, and reinforcement learning from human feedback.

## Learning Objectives

By the end of this week, you will be able to:
- Apply knowledge distillation to compress models
- Implement and understand quantization techniques
- Explain the RLHF pipeline and its components
- Understand DPO as an alternative to RLHF
- Analyze trade-offs between model size and performance
- Train a simple preference model

## Core Topics

### 1. Model Compression Overview

**Why Compress?**
- Reduce inference cost
- Enable edge deployment
- Faster iteration
- Environmental impact

**Compression Techniques**
```
                    ┌─────────────────┐
                    │   Compression   │
                    └────────┬────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌────────────┐   ┌─────────────┐   ┌────────────┐
    │ Distillation│   │ Quantization│   │  Pruning   │
    └────────────┘   └─────────────┘   └────────────┘
```

### 2. Knowledge Distillation

**Key Idea**: Train small "student" to mimic large "teacher"

**Soft Labels**
```python
# Teacher produces soft probabilities
teacher_logits = teacher(x)
soft_labels = F.softmax(teacher_logits / temperature, dim=-1)

# Student learns from soft labels
student_logits = student(x)
student_probs = F.log_softmax(student_logits / temperature, dim=-1)

# KL divergence loss
distill_loss = F.kl_div(student_probs, soft_labels, reduction='batchmean')
```

**Combined Loss**
```python
# Often combine with hard labels
alpha = 0.5
total_loss = alpha * distill_loss + (1 - alpha) * F.cross_entropy(student_logits, hard_labels)
```

**DistilBERT** (Sanh et al., 2019)
- 40% smaller than BERT
- 60% faster
- Retains 97% of performance
- Trained on masked LM + distillation

**Distillation Strategies**:
- **Output distillation**: Match final logits
- **Intermediate layer distillation**: Match hidden states
- **Attention distillation**: Match attention patterns

### 3. Quantization

**Goal**: Use fewer bits for weights/activations

**Data Types**
```
FP32:  ████████████████████████████████  (32 bits)
FP16:  ████████████████                  (16 bits)
INT8:  ████████                          (8 bits)
INT4:  ████                              (4 bits)
```

**Post-Training Quantization (PTQ)**
```python
import torch

# Simple INT8 quantization
def quantize_tensor(tensor, num_bits=8):
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, qmin, qmax).to(torch.int8)

    return q_tensor, scale, zero_point
```

**Quantization-Aware Training (QAT)**
- Simulate quantization during training
- Model learns to be robust to quantization errors
- Better quality than PTQ

**QLoRA** (Dettmers et al., 2023)
- 4-bit quantized base model
- LoRA adapters in full precision
- Enables fine-tuning 65B models on single GPU

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)
```

### 4. Pruning

**Goal**: Remove unnecessary weights

**Types**:
- **Unstructured**: Individual weights to zero
- **Structured**: Remove entire neurons/heads

**Lottery Ticket Hypothesis** (Frankle & Carbin, 2019)
- Dense networks contain sparse subnetworks
- These "winning tickets" match full network performance
- Key: need right initialization

### 5. Reinforcement Learning from Human Feedback (RLHF)

**The Alignment Problem**
- Pre-trained LMs don't know what humans want
- Need to align model behavior with human preferences

**RLHF Pipeline**
```
Step 1: Supervised Fine-Tuning (SFT)
        Train on human demonstrations

Step 2: Reward Model Training
        Learn to predict human preferences

Step 3: RL Fine-Tuning
        Optimize policy with reward model
```

**Step 1: SFT**
```python
# Train on demonstration data
for prompt, response in demonstrations:
    loss = model.forward(prompt + response)
    loss.backward()
```

**Step 2: Reward Model**
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.base = base_model
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden = self.base(x).last_hidden_state[:, -1]
        return self.reward_head(hidden)

# Training: Bradley-Terry model
def preference_loss(reward_chosen, reward_rejected):
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()
```

**Step 3: PPO Training**
```
For each batch of prompts:
    1. Generate responses with current policy
    2. Score with reward model
    3. Compute PPO loss with KL penalty
    4. Update policy
```

**KL Penalty**
```python
reward_with_kl = reward - beta * kl_divergence(policy, reference_policy)
```
- Prevents policy from deviating too far from SFT model
- Maintains response quality

### 6. Direct Preference Optimization (DPO)

**Key Insight**: Skip the reward model entirely

**DPO Loss**
```python
def dpo_loss(policy_chosen_logprobs, policy_rejected_logprobs,
             ref_chosen_logprobs, ref_rejected_logprobs, beta):

    chosen_logratios = policy_chosen_logprobs - ref_chosen_logprobs
    rejected_logratios = policy_rejected_logprobs - ref_rejected_logprobs

    loss = -F.logsigmoid(beta * (chosen_logratios - rejected_logratios))
    return loss.mean()
```

**Advantages**:
- No separate reward model
- No RL training (simpler, more stable)
- Directly optimizes the same objective

**Comparison**

| Aspect | RLHF | DPO |
|--------|------|-----|
| Complexity | High | Low |
| Training | RL (unstable) | Supervised |
| Memory | Need reward + policy | Just policy |
| Performance | Strong | Comparable |

### 7. RLAIF (RL from AI Feedback)

**Key Idea**: Use AI to generate preference labels

```python
def get_ai_preference(response_a, response_b, prompt):
    evaluation_prompt = f"""
    Given the prompt: {prompt}

    Response A: {response_a}
    Response B: {response_b}

    Which response is better? Output only 'A' or 'B'.
    """
    return evaluator_model.generate(evaluation_prompt)
```

**Advantages**:
- Cheaper than human annotation
- Scales better
- Can iterate faster

**Challenges**:
- AI biases propagate
- May not capture human preferences
- Evaluation quality varies

## Key References

### Required Reading

1. **Sanh et al. (2019) - "DistilBERT, a distilled version of BERT"**
   - Knowledge distillation for BERT
   - [Paper](https://arxiv.org/abs/1910.01108)

2. **Ouyang et al. (2022) - "Training language models to follow instructions with human feedback"**
   - InstructGPT/RLHF paper
   - [Paper](https://arxiv.org/abs/2203.02155)

3. **Rafailov et al. (2023) - "Direct Preference Optimization"**
   - DPO paper
   - [Paper](https://arxiv.org/abs/2305.18290)

### Recommended Reading

4. **Dettmers et al. (2023) - "QLoRA: Efficient Finetuning of Quantized LLMs"**
   - 4-bit quantization with LoRA
   - [Paper](https://arxiv.org/abs/2305.14314)

5. **Frankle & Carlin (2019) - "The Lottery Ticket Hypothesis"**
   - Sparse networks in dense networks
   - [Paper](https://arxiv.org/abs/1803.03635)

6. **Lee et al. (2023) - "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"**
   - AI feedback for RL
   - [Paper](https://arxiv.org/abs/2309.00267)

## Practical Exercise

### Exercise 1: Knowledge Distillation

**Objective**: Distill BERT into a smaller model

```python
class DistillationTrainer:
    def __init__(self, teacher, student, temperature=2.0, alpha=0.5):
        self.teacher = teacher.eval()
        self.student = student
        self.temp = temperature
        self.alpha = alpha

    def distillation_loss(self, student_logits, teacher_logits, labels):
        # Soft label loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temp, dim=-1),
            F.softmax(teacher_logits / self.temp, dim=-1),
            reduction='batchmean'
        ) * (self.temp ** 2)

        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

**Tasks**:
1. Train student on GLUE task
2. Compare with training from scratch
3. Measure inference speedup

### Exercise 2: Quantization Experiment

**Objective**: Apply quantization and measure impact

```python
from transformers import AutoModelForCausalLM
import torch

# Load model in different precisions
model_fp32 = AutoModelForCausalLM.from_pretrained("gpt2")
model_fp16 = model_fp32.half()

# Measure
def benchmark(model, input_ids, n_runs=100):
    # Measure latency and memory
    pass
```

| Precision | Memory | Latency | Perplexity |
|-----------|--------|---------|------------|
| FP32 | | | |
| FP16 | | | |
| INT8 | | | |

### Exercise 3: Simple RLHF/DPO

**Objective**: Train a preference model and apply DPO

**Part A: Preference Data**
```python
# Load or create preference pairs
preferences = [
    {"prompt": "...", "chosen": "...", "rejected": "..."},
    ...
]
```

**Part B: Train with DPO**
```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=train_dataset,
)
trainer.train()
```

### Exercise Files
- [exercises/distillation.py](./exercises/distillation.py)
- [exercises/quantization_benchmark.py](./exercises/quantization_benchmark.py)
- [exercises/dpo_training.py](./exercises/dpo_training.py)

## Study Questions

1. Why do soft labels provide more information than hard labels?
2. What causes accuracy loss during quantization?
3. Why is a KL penalty needed in RLHF?
4. How does DPO avoid the reward model?
5. What are the limitations of RLAIF?

## Additional Resources

### Libraries
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - Quantization
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning

### Tutorials
- [Hugging Face DPO Tutorial](https://huggingface.co/docs/trl/dpo_trainer)
- [QLoRA Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

## Next Week Preview

Week 11 focuses on debugging and interpreting neural NLP models:
- Probing classifiers
- Attention analysis
- Mechanistic interpretability
- Model editing

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 11, ensure you can:
- [ ] Implement knowledge distillation
- [ ] Apply quantization to models
- [ ] Explain the RLHF pipeline
- [ ] Train a model with DPO
- [ ] Analyze size/performance trade-offs
