# Week 7: Instruction Tuning & Efficient Fine-Tuning

## Overview

This week explores how to adapt pre-trained language models to follow instructions and perform specific tasks. We'll cover the spectrum from few-shot prompting to full fine-tuning, with emphasis on parameter-efficient methods like LoRA.

## Learning Objectives

By the end of this week, you will be able to:
- Compare few-shot prompting with fine-tuning approaches
- Understand instruction tuning and its benefits
- Implement LoRA for parameter-efficient fine-tuning
- Apply adapter modules and prompt tuning
- Fine-tune models for question answering tasks
- Evaluate trade-offs between different adaptation methods

## Core Topics

### 1. The Adaptation Spectrum

```
Zero-shot → Few-shot → Prompt Tuning → Adapters → LoRA → Full Fine-tuning
    ↑                                                              ↑
No training                                               All parameters
```

**Trade-offs**:
- **Compute**: Zero-shot needs none; full fine-tuning needs significant GPU
- **Data**: Few-shot needs examples; fine-tuning needs datasets
- **Flexibility**: Prompting is flexible; fine-tuning is task-specific
- **Performance**: Generally improves left to right (with enough data)

### 2. Few-Shot In-Context Learning

**GPT-3's Breakthrough** (Brown et al., 2020)
- Large models can learn from examples in the prompt
- No gradient updates needed
- Performance scales with model size

**Example**:
```
Translate English to French:

English: sea otter
French: loutre de mer

English: cheese
French: fromage

English: artificial intelligence
French:
```

**Limitations**:
- Context window limits number of examples
- Sensitive to example ordering
- Not all tasks work well with prompting
- Can't learn truly new behaviors

### 3. Instruction Tuning

**Key Idea**: Fine-tune on diverse instruction-following tasks

**FLAN (Wei et al., 2021)**
- Fine-tuned Language Net
- Train on 60+ NLP tasks phrased as instructions
- Improves zero-shot performance on unseen tasks

**Instruction Format**:
```
Task: Determine if the sentiment is positive or negative.
Input: This movie was absolutely terrible.
Output: negative
```

**Benefits**:
- Better zero-shot generalization
- More consistent following of instructions
- Enables new capabilities

### 4. Parameter-Efficient Fine-Tuning (PEFT)

**Motivation**:
- Full fine-tuning of 7B+ models is expensive
- Storing separate copies for each task is impractical
- Small updates can achieve similar performance

### 5. LoRA (Low-Rank Adaptation)

**Key Insight**: Weight updates have low intrinsic rank

**Method**:
```
W' = W + ΔW = W + BA

where:
- W: frozen pre-trained weights [d × k]
- B: low-rank matrix [d × r]
- A: low-rank matrix [r × k]
- r << min(d, k) (typically r = 8, 16, 64)
```

**Implementation**:
```python
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False

        d_out, d_in = original_layer.weight.shape
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out
```

**Benefits**:
- 0.1-1% of trainable parameters
- No additional inference latency (merge weights)
- Easy to swap task-specific adapters
- Works surprisingly well

**Using PEFT Library**:
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 0.1% of 7B
```

### 6. Adapter Modules

**Architecture**: Small bottleneck layers inserted into transformer

```
          ┌─────────────────────┐
          │   Feed-Forward      │
          └─────────┬───────────┘
                    │
          ┌─────────▼───────────┐
          │   Adapter Down      │  (d → r)
          │   Non-linearity     │
          │   Adapter Up        │  (r → d)
          └─────────┬───────────┘
                    │
              ┌─────▼─────┐
              │  Residual │
              └───────────┘
```

**Properties**:
- 1-5% of parameters
- Slight inference overhead
- Can be combined with LoRA

### 7. Prompt Tuning

**Soft Prompts**: Learn continuous token embeddings

```python
class PromptTuning(nn.Module):
    def __init__(self, model, num_tokens, embed_dim):
        self.soft_prompt = nn.Parameter(torch.randn(num_tokens, embed_dim))
        self.model = model  # frozen

    def forward(self, input_ids):
        input_embeds = self.model.embed_tokens(input_ids)
        # Prepend soft prompt
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        return self.model(inputs_embeds=full_embeds)
```

**Properties**:
- Only prompt tokens are trained
- Works better with larger models
- Very parameter-efficient

### 8. Comparison Summary

| Method | Trainable Params | Memory | Inference | Performance |
|--------|-----------------|--------|-----------|-------------|
| Full Fine-tuning | 100% | High | Same | Best |
| LoRA | 0.1-1% | Low | Same* | Very Good |
| Adapters | 1-5% | Low | Slight overhead | Good |
| Prompt Tuning | 0.01% | Minimal | Same | Good (large models) |
| Few-shot | 0% | None | Same | Variable |

*LoRA weights can be merged after training

## Key References

### Required Reading

1. **Brown et al. (2020) - "Language Models are Few-Shot Learners"**
   - GPT-3 paper introducing in-context learning
   - [Paper](https://arxiv.org/abs/2005.14165)

2. **Wei et al. (2021) - "Finetuned Language Models Are Zero-Shot Learners"**
   - FLAN paper on instruction tuning
   - [Paper](https://arxiv.org/abs/2109.01652)

3. **Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"**
   - LoRA methodology
   - [Paper](https://arxiv.org/abs/2106.09685)

### Recommended Reading

4. **Lester et al. (2021) - "The Power of Scale for Parameter-Efficient Prompt Tuning"**
   - Soft prompt tuning
   - [Paper](https://arxiv.org/abs/2104.08691)

5. **Houlsby et al. (2019) - "Parameter-Efficient Transfer Learning for NLP"**
   - Adapter modules
   - [Paper](https://arxiv.org/abs/1902.00751)

6. **Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units"**
   - BPE for NMT (context for tokenization)
   - [Paper](https://arxiv.org/abs/1508.07909)

## Practical Exercise

### Exercise 1: Fine-tune with LoRA

**Objective**: Fine-tune a model for QA using LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

# Train on SQuAD or similar
dataset = load_dataset("squad")
# ... training loop
```

**Tasks**:
1. Compare training time with full fine-tuning
2. Evaluate on held-out test set
3. Merge LoRA weights and compare inference

### Exercise 2: Compare PEFT Methods

**Objective**: Compare LoRA, adapters, and prompt tuning

| Method | Train Time | Memory | Test Accuracy |
|--------|-----------|--------|---------------|
| Full FT | | | |
| LoRA r=8 | | | |
| LoRA r=64 | | | |
| Adapters | | | |
| Prompt Tuning | | | |

### Exercise 3: Instruction Tuning

**Objective**: Format data as instructions and fine-tune

```python
def format_instruction(example):
    return f"""### Instruction:
Answer the following question based on the context.

### Context:
{example['context']}

### Question:
{example['question']}

### Answer:
{example['answer']}"""
```

**Tasks**:
1. Convert QA dataset to instruction format
2. Fine-tune with LoRA
3. Evaluate zero-shot on new question types

### Exercise Files
- [exercises/lora_finetuning.py](./exercises/lora_finetuning.py)
- [exercises/peft_comparison.py](./exercises/peft_comparison.py)
- [exercises/instruction_formatting.py](./exercises/instruction_formatting.py)

## Study Questions

1. Why does LoRA work despite training such few parameters?
2. When would you prefer full fine-tuning over PEFT methods?
3. How does instruction tuning improve zero-shot generalization?
4. What's the relationship between LoRA rank and model capacity?
5. Why does prompt tuning work better with larger models?

## Additional Resources

### Libraries
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

### Tutorials
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Tutorial](https://huggingface.co/blog/lora)

### Pre-trained Models
- FLAN-T5 (instruction-tuned T5)
- Alpaca (instruction-tuned LLaMA)
- Vicuna (chat fine-tuned LLaMA)

## Next Week Preview

Week 8 focuses on experimental design and human annotation:
- Designing NLP experiments properly
- Data collection best practices
- Inter-annotator agreement metrics

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 8, ensure you can:
- [ ] Explain the LoRA algorithm mathematically
- [ ] Fine-tune a model using PEFT library
- [ ] Compare efficiency of different PEFT methods
- [ ] Format data for instruction tuning
- [ ] Evaluate when to use each adaptation method
