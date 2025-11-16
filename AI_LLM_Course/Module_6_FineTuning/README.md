# Module 6: Fine-tuning & Optimization

## Overview

Master the art of fine-tuning Large Language Models for specific tasks. Learn full fine-tuning, parameter-efficient methods (LoRA, QLoRA), RLHF, quantization, and production deployment.

**Duration**: 6-8 weeks (140-200 hours)
**Difficulty**: Advanced

---

## Learning Objectives

✅ Fine-tune LLMs using full fine-tuning and PEFT methods
✅ Implement LoRA and QLoRA for efficient training
✅ Apply RLHF (Reinforcement Learning from Human Feedback)
✅ Quantize models to 4-bit and 8-bit
✅ Adapt models to specific domains (medical, legal, code)
✅ Deploy fine-tuned models to production
✅ Optimize training for cost and speed
✅ Track experiments and manage model versions

---

## Weekly Breakdown

### Week 28: Fine-tuning Fundamentals (20-25 hours)

**Topics**:
- Transfer learning principles
- Full fine-tuning process
- Dataset preparation and formatting
- Training loop implementation
- Hyperparameter selection
- Loss monitoring and early stopping

**Code Examples**: 25+
**Project**: Fine-tune GPT-2 on custom dataset
**Key Skills**: Dataset curation, training pipelines

---

### Week 29: Instruction Tuning (18-22 hours)

**Topics**:
- Instruction datasets (Alpaca, Dolly, etc.)
- Supervised Fine-Tuning (SFT)
- Instruction formatting best practices
- Multi-task instruction learning
- Evaluation of instruction-following

**Code Examples**: 20+
**Project**: Create instruction dataset and fine-tune model
**Key Skills**: Prompt engineering, instruction design

**Example Code**:
```python
# Instruction format
instruction_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
"""

# Fine-tuning with instructions
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=instruction_dataset,
    dataset_text_field="text",
)

trainer.train()
```

---

### Week 30: Parameter-Efficient Fine-Tuning (PEFT) (22-28 hours)

**Topics**:
- LoRA (Low-Rank Adaptation) theory
- QLoRA (Quantized LoRA)
- Adapter layers
- Prefix tuning and P-tuning
- Comparing PEFT methods
- Memory and compute savings

**Code Examples**: 30+
**Project**: Compare 6 different PEFT methods
**Key Skills**: Efficient training, adapter management

**LoRA Example**:
```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

# Wrap model with LoRA
model = get_peft_model(base_model, lora_config)

# Only 0.1% of parameters are trainable!
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.237
```

**QLoRA Example**:
```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Apply LoRA on top
model = get_peft_model(model, lora_config)

# Train 7B model on single GPU!
```

---

### Week 31: Advanced Fine-tuning Techniques (20-25 hours)

**Topics**:
- Mixed precision training (FP16, BF16)
- Gradient accumulation strategies
- Gradient checkpointing
- DeepSpeed integration (ZeRO stages)
- Multi-GPU and distributed training
- Memory optimization techniques

**Code Examples**: 25+
**Project**: Scale training to multiple GPUs
**Key Skills**: Distributed training, optimization

**DeepSpeed Example**:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    deepspeed="ds_config.json",  # DeepSpeed config
    fp16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
)

# ds_config.json
{
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

---

### Week 32: RLHF (Reinforcement Learning from Human Feedback) (25-30 hours)

**Topics**:
- Reward modeling
- PPO (Proximal Policy Optimization)
- DPO (Direct Preference Optimization)
- Constitutional AI principles
- Human preference data collection
- Alignment and safety

**Code Examples**: 20+
**Project**: Implement simple RLHF pipeline
**Key Skills**: RL for LLMs, alignment

**RLHF Pipeline**:
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model

# 1. Load model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ref_model = create_reference_model(model)

# 2. Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "reward_model_checkpoint"
)

# 3. PPO configuration
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=128,
)

# 4. Create PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 5. Training loop
for epoch in range(ppo_config.num_epochs):
    for batch in dataloader:
        query_tensors = batch["input_ids"]

        # Generate responses
        response_tensors = ppo_trainer.generate(query_tensors)

        # Compute rewards
        rewards = [reward_model(response) for response in response_tensors]

        # PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

**DPO Example (simpler alternative)**:
```python
from trl import DPOTrainer

# DPO is simpler than PPO - no reward model needed!
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,  # DPO temperature
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

---

### Week 33: Quantization & Compression (18-22 hours)

**Topics**:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- 4-bit, 8-bit quantization
- GPTQ, AWQ methods
- Pruning techniques
- Knowledge distillation

**Code Examples**: 20+
**Project**: Compress model to 4-bit while maintaining 95%+ performance
**Key Skills**: Model compression, deployment optimization

**Quantization Examples**:
```python
# Method 1: BitsAndBytes (easiest)
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=config,
)

# Method 2: GPTQ (better compression)
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    use_safetensors=True,
    device="cuda:0",
)

# Method 3: AWQ (faster inference)
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,
)

# Knowledge Distillation
from transformers import DistillationTrainer

teacher_model = AutoModelForCausalLM.from_pretrained("large_model")
student_model = AutoModelForCausalLM.from_pretrained("small_model")

distillation_trainer = DistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    temperature=2.0,
    alpha=0.5,  # Weight for distillation loss
)
```

---

### Week 34: Domain Adaptation (22-28 hours)

**Topics**:
- Medical domain fine-tuning
- Legal domain adaptation
- Code generation specialization
- Catastrophic forgetting prevention
- Continual learning strategies
- Domain-specific evaluation

**Code Examples**: 25+
**Projects**: 3 domain-specific models (medical, legal, code)
**Key Skills**: Domain expertise, specialized training

**Domain-Specific Fine-tuning**:
```python
# Medical domain example
medical_dataset = load_dataset("medical_qa_dataset")

# Use domain-specific tokenizer vocabulary
tokenizer.add_tokens([
    "diagnosis", "treatment", "symptom", "medication",
    "cardiovascular", "pulmonary", "neurological"
])

model.resize_token_embeddings(len(tokenizer))

# Domain-specific training config
training_args = TrainingArguments(
    # Longer training for domain adaptation
    num_train_epochs=10,
    # Lower learning rate to preserve general knowledge
    learning_rate=1e-5,
    # Domain-specific evaluation
    eval_strategy="steps",
    eval_steps=500,
)

# Prevent catastrophic forgetting
from torch.nn import functional as F

def compute_loss_with_replay(model, batch, replay_batch):
    # Loss on new domain data
    outputs = model(**batch)
    loss_new = outputs.loss

    # Loss on general domain data (replay)
    replay_outputs = model(**replay_batch)
    loss_replay = replay_outputs.loss

    # Combined loss
    return 0.7 * loss_new + 0.3 * loss_replay
```

---

### Week 35: Production Fine-tuning (25-30 hours)

**Topics**:
- Hyperparameter optimization (Grid, Random, Bayesian)
- Experiment tracking (W&B, MLflow)
- Model versioning and registry
- A/B testing fine-tuned models
- Cost optimization strategies
- Monitoring and debugging training

**Code Examples**: 30+
**Project**: Complete MLOps pipeline for LLM fine-tuning
**Key Skills**: MLOps, production deployment

**Experiment Tracking**:
```python
import wandb
from transformers import TrainingArguments

# Initialize W&B
wandb.init(project="llm-finetuning", name="llama2-medical")

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # Auto-log to W&B
    logging_steps=10,
)

# Log custom metrics
wandb.log({
    "custom_metric": value,
    "learning_rate": lr,
})

# Hyperparameter sweep
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'eval/loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 1e-6, 'max': 1e-4},
        'per_device_train_batch_size': {'values': [4, 8, 16]},
        'lora_r': {'values': [8, 16, 32]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="llm-finetuning")
wandb.agent(sweep_id, function=train_fn, count=20)
```

**Model Registry**:
```python
# Save model with metadata
model.save_pretrained("./final_model")

metadata = {
    "model_name": "llama2-medical-v1",
    "base_model": "meta-llama/Llama-2-7b-hf",
    "dataset": "medical_qa_100k",
    "training_time": "4 hours",
    "eval_accuracy": 0.92,
    "parameters": {
        "lora_r": 16,
        "learning_rate": 2e-5,
    }
}

# Upload to HuggingFace Hub
model.push_to_hub("your-org/llama2-medical-v1")

# Or MLflow
import mlflow

mlflow.transformers.log_model(
    transformers_model=model,
    artifact_path="model",
    registered_model_name="llama2-medical",
)
```

---

## Key Projects

### Project 1: Custom Domain Chatbot
Fine-tune LLaMA-2 7B for customer support:
- Collect and format conversation data
- Apply LoRA fine-tuning
- Implement RLHF for better responses
- Deploy with FastAPI
- Monitor performance

**Time**: 30-40 hours

### Project 2: Code Assistant
Fine-tune for code generation:
- Use GitHub code dataset
- Implement code-specific evaluation
- Add function calling
- Deploy with Gradio interface

**Time**: 25-30 hours

### Project 3: Specialized Medical Assistant
Fine-tune for medical Q&A:
- Medical literature dataset
- Domain adaptation techniques
- Safety and hallucination prevention
- HIPAA-compliant deployment

**Time**: 30-40 hours

---

## Tools & Frameworks

- **PyTorch** + **Transformers**
- **PEFT** (LoRA, QLoRA, Adapters)
- **TRL** (RLHF, PPO, DPO)
- **BitsAndBytes** (Quantization)
- **DeepSpeed** (Distributed Training)
- **Weights & Biases** (Experiment Tracking)
- **HuggingFace Hub** (Model Hosting)

---

## Assessment

### Skills Checklist
- [ ] Can fine-tune GPT-2 and LLaMA models
- [ ] Understand and implement LoRA/QLoRA
- [ ] Can apply RLHF or DPO
- [ ] Quantize models to 4/8-bit
- [ ] Deploy fine-tuned models to production
- [ ] Track experiments systematically
- [ ] Optimize training costs
- [ ] Handle multi-GPU training

### Self-Assessment (1-5)
- Full fine-tuning: ___/5
- PEFT methods: ___/5
- RLHF: ___/5
- Quantization: ___/5
- Production deployment: ___/5

**Goal**: All areas 4+

---

**Next**: [Module 7: Advanced Topics & Projects](../Module_7_Advanced/README.md)

---

*Duration: 6-8 weeks*
*Code Examples: 165+*
*Projects: 3 production applications*
