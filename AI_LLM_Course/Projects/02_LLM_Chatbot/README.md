# Capstone Project: Custom Domain LLM Chatbot

## Project Overview

Build a production-ready chatbot by fine-tuning LLaMA-2 7B for a specific domain (customer support, medical, legal, or technical).

**Duration**: 30-40 hours
**Difficulty**: Advanced
**Skills**: LLM Fine-tuning, LoRA, RAG, Deployment

---

## Objectives

✅ Collect and format domain-specific conversation data
✅ Fine-tune LLaMA-2 7B using LoRA
✅ Implement RAG for knowledge retrieval
✅ Apply RLHF for better responses
✅ Build FastAPI backend
✅ Create web interface
✅ Deploy to cloud
✅ Monitor performance

---

## Architecture

```
User → Web UI → FastAPI → RAG System → Fine-tuned LLM → Response
                     ↓
                Vector DB (ChromaDB)
                     ↓
                Knowledge Base
```

---

## Tech Stack

- **LLM**: LLaMA-2 7B
- **Fine-tuning**: PEFT (LoRA)
- **Vector DB**: ChromaDB
- **Backend**: FastAPI
- **Frontend**: Streamlit or Gradio
- **Deployment**: Docker + AWS/GCP
- **Monitoring**: Weights & Biases

---

## Phase 1: Data Collection & Preparation (6-8 hours)

### Tasks
1. Collect domain-specific conversations
2. Format data for instruction tuning
3. Create train/validation splits
4. Quality control

### Data Format

```python
# Instruction format
{
    "instruction": "User query or question",
    "input": "Additional context (optional)",
    "output": "Expected response"
}

# Example - Customer Support
{
    "instruction": "How do I reset my password?",
    "input": "",
    "output": "To reset your password:\\n1. Go to the login page\\n2. Click 'Forgot Password'\\n3. Enter your email\\n4. Check your email for reset link\\n5. Follow the instructions\\n\\nIf you need further assistance, contact support@company.com"
}
```

### Data Collection Script

```python
import json
import pandas as pd

class ConversationDataset:
    def __init__(self, domain='customer_support'):
        self.domain = domain
        self.conversations = []

    def load_from_csv(self, filepath):
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            self.conversations.append({
                'instruction': row['question'],
                'input': row.get('context', ''),
                'output': row['answer']
            })

    def format_for_training(self, template='alpaca'):
        if template == 'alpaca':
            formatted = []
            for conv in self.conversations:
                text = f\"\"\"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{conv['instruction']}

### Response:
{conv['output']}\"\"\"
                formatted.append({'text': text})
            return formatted

    def save(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.conversations, f, indent=2)

# Usage
dataset = ConversationDataset(domain='customer_support')
dataset.load_from_csv('data/support_qa.csv')
formatted = dataset.format_for_training()
```

**Deliverable**: `data/training_data.json` with 1000+ examples

---

## Phase 2: Fine-tuning with LoRA (8-10 hours)

### Setup

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=\"nf4\",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    \"meta-llama/Llama-2-7b-hf\",
    quantization_config=bnb_config,
    device_map=\"auto\",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(\"meta-llama/Llama-2-7b-hf\")
tokenizer.pad_token = tokenizer.eos_token
```

### LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=[
        \"q_proj\",
        \"k_proj\",
        \"v_proj\",
        \"o_proj\",
        \"gate_proj\",
        \"up_proj\",
        \"down_proj\",
    ],
    lora_dropout=0.05,
    bias=\"none\",
    task_type=\"CAUSAL_LM\"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 6,738,415,616 || trainable%: 0.62%
```

### Training

```python
training_args = TrainingArguments(
    output_dir=\"./results\",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy=\"steps\",
    save_total_limit=3,
    push_to_hub=False,
    report_to=\"wandb\",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    dataset_text_field=\"text\",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# Train!
trainer.train()

# Save
model.save_pretrained(\"./fine_tuned_model\")
tokenizer.save_pretrained(\"./fine_tuned_model\")
```

**Deliverable**: Fine-tuned model with eval loss < 1.0

---

## Phase 3: RAG Implementation (6-8 hours)

### Knowledge Base Setup

```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load documents
loader = DirectoryLoader('data/knowledge_base/', glob=\"**/*.txt\")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=\"sentence-transformers/all-MiniLM-L6-v2\"
)

# Create vector store
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=\"./chroma_db\"
)
vectordb.persist()
```

### RAG Pipeline

```python
class RAGChatbot:
    def __init__(self, model, tokenizer, vectordb):
        self.model = model
        self.tokenizer = tokenizer
        self.vectordb = vectordb

    def retrieve_context(self, query, k=3):
        docs = self.vectordb.similarity_search(query, k=k)
        context = \"\\n\\n\".join([doc.page_content for doc in docs])
        return context

    def generate_response(self, query, max_length=256):
        # Retrieve relevant context
        context = self.retrieve_context(query)

        # Format prompt
        prompt = f\"\"\"Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:\"\"\"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors=\"pt\").to(\"cuda\")

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split(\"Answer:\")[-1].strip()

# Usage
chatbot = RAGChatbot(model, tokenizer, vectordb)
response = chatbot.generate_response(\"How do I reset my password?\")
print(response)
```

---

## Phase 4: Backend API (4-6 hours)

### FastAPI Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title=\"Domain Chatbot API\")

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: list = []

# Load model and chatbot
chatbot = load_chatbot()  # Your loading function

@app.post(\"/chat\", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = chatbot.generate_response(request.message)

        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id or generate_id(),
            sources=[]  # Add source documents if needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(\"/health\")
async def health():
    return {\"status\": \"healthy\"}

if __name__ == \"__main__\":
    uvicorn.run(app, host=\"0.0.0.0\", port=8000)
```

---

## Phase 5: Frontend UI (3-4 hours)

### Streamlit Interface

```python
import streamlit as st
import requests

st.title(\"🤖 Domain Expert Chatbot\")

# Initialize chat history
if \"messages\" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message[\"role\"]):
        st.markdown(message[\"content\"])

# Chat input
if prompt := st.chat_input(\"Ask me anything...\"):
    # Add user message
    st.session_state.messages.append({\"role\": \"user\", \"content\": prompt})
    with st.chat_message(\"user\"):
        st.markdown(prompt)

    # Get response from API
    response = requests.post(
        \"http://localhost:8000/chat\",
        json={\"message\": prompt}
    )

    bot_response = response.json()[\"response\"]

    # Add assistant response
    st.session_state.messages.append({\"role\": \"assistant\", \"content\": bot_response})
    with st.chat_message(\"assistant\"):
        st.markdown(bot_response)
```

---

## Phase 6: Deployment (4-6 hours)

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - \"8000:8000\"
    environment:
      - MODEL_PATH=/models
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Evaluation Criteria

### Technical (80 points)
- [ ] Fine-tuned model deployed (20 points)
- [ ] RAG working correctly (15 points)
- [ ] API functional (15 points)
- [ ] UI user-friendly (10 points)
- [ ] Response quality (10 points)
- [ ] Response time < 3s (10 points)

### Documentation (20 points)
- [ ] Setup instructions (5 points)
- [ ] Architecture diagram (5 points)
- [ ] API documentation (5 points)
- [ ] Demo video (5 points)

**Total: 100 points**

---

## Expected Metrics

- **Response Time**: < 3 seconds
- **Answer Relevance**: > 85%
- **Factual Accuracy**: > 90%
- **User Satisfaction**: > 4/5

---

## Submission

1. GitHub repository with code
2. Fine-tuned model (HuggingFace Hub)
3. Demo video (5 minutes)
4. Documentation
5. Deployed link (optional)

---

**Duration**: 30-40 hours
**Level**: Advanced
**Tech**: PyTorch, Transformers, FastAPI, Docker

**Good luck! 🚀**
