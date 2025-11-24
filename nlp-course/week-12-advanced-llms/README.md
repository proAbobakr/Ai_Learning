# Week 12: Advanced LLMs, Agents & Long Contexts

## Overview

This week covers the frontier of large language models: modern architectures like LLaMA, GPT-4, and Claude, techniques for handling long contexts, and how LLMs can be augmented with tool use to become agents.

## Learning Objectives

By the end of this week, you will be able to:
- Understand architectural differences among modern LLMs
- Implement long-context techniques (RoPE, FlashAttention)
- Build LLM agents with tool use
- Apply ReAct prompting for reasoning
- Handle context length limitations
- Evaluate trade-offs in agent design

## Core Topics

### 1. Modern LLM Landscape

**Key Models (as of 2024)**

| Model | Params | Context | Architecture | Access |
|-------|--------|---------|--------------|--------|
| GPT-4 | ~1.8T* | 128K | MoE | API |
| Claude 3 | - | 200K | - | API |
| LLaMA 3 | 8-70B | 8K | Dense | Open weights |
| Mistral | 7B | 32K | Dense + SWA | Open |
| Mixtral | 8x7B | 32K | MoE | Open |

*Estimated, not confirmed

**Architectural Innovations**:

**Mixture of Experts (MoE)**
```
Input → Router → Selected Experts → Output
                    ↓
            Only k of N experts active
```
- Sparse activation: more params, same compute
- Router network selects experts per token
- Enables larger models with manageable cost

**Grouped Query Attention (GQA)**
- Reduce KV cache memory
- Share K/V heads across Q heads
- LLaMA 2 70B uses 8 KV heads for 64 Q heads

### 2. Long Context Techniques

**Challenge**: Self-attention is O(n²) in sequence length

**Rotary Position Embedding (RoPE)**
```python
def apply_rope(x, positions, dim):
    # Rotate pairs of dimensions based on position
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    angles = positions.unsqueeze(-1) * freqs
    cos, sin = angles.cos(), angles.sin()

    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

**Properties**:
- Encodes relative positions
- Extrapolates to longer sequences
- No absolute position limit

**FlashAttention** (Dao et al., 2022)
```
Standard:  O(n²) memory, many HBM reads
Flash:     O(n) memory, tiled computation
```

Key Ideas:
- Compute attention in tiles
- Keep intermediate results in SRAM
- Never materialize full attention matrix

```python
# Using Flash Attention
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

**Sliding Window Attention**
```
Full:    [████████████████]  Every token attends to all
Sliding: [████░░░░░░░░░░░░]  Window of w tokens
                ↓
         Layers stack to increase effective context
```
- Used by Mistral
- O(n × w) instead of O(n²)
- Stack layers for long-range

**ALiBi (Attention with Linear Biases)**
- Add linear bias based on distance
- No learned position embeddings
- Extrapolates well

### 3. Context Length Extension

**Position Interpolation**
```python
# If trained on 4K, extend to 16K
scale = original_max_len / target_max_len
position_ids = position_ids * scale
```

**YaRN (Yet another RoPE extensioN)**
- Improved interpolation
- Better preserves learned patterns
- Enables 128K+ contexts

**Landmark Attention**
- Select important tokens as landmarks
- Full attention to landmarks
- Compressed attention otherwise

### 4. LLM Agents

**Definition**: LLM + Tools + Memory + Planning

**Basic Agent Loop**
```python
def agent_loop(user_query, tools, max_steps=10):
    context = [{"role": "user", "content": user_query}]

    for step in range(max_steps):
        response = llm.generate(context)

        if response.wants_to_use_tool():
            tool_name, tool_args = response.parse_tool_call()
            tool_result = tools[tool_name](**tool_args)
            context.append({"role": "tool", "content": tool_result})
        else:
            return response.final_answer()

    return "Max steps reached"
```

**Tool Types**:
- **Retrieval**: Search documents, web
- **Code execution**: Python, SQL
- **APIs**: Weather, calculator, search
- **Actions**: Send email, create file

### 5. ReAct: Reasoning + Acting

**Pattern**: Think → Act → Observe → Repeat

```
User: What's the weather in Paris and should I bring an umbrella?

Thought: I need to find the current weather in Paris
Action: weather_api(location="Paris")
Observation: {"temp": 15, "condition": "rainy", "precipitation": 80%}

Thought: It's rainy with 80% precipitation, so umbrella recommended
Action: none
Answer: It's currently 15°C and rainy in Paris with 80% chance of
        precipitation. Yes, you should definitely bring an umbrella!
```

**Implementation**:
```python
REACT_PROMPT = """You have access to the following tools:
{tool_descriptions}

Use this format:
Thought: [your reasoning]
Action: tool_name(args)
Observation: [tool output will appear here]
... (repeat as needed)
Thought: I have enough information
Answer: [final response]

Question: {question}
"""

def react_agent(question, tools):
    prompt = REACT_PROMPT.format(
        tool_descriptions=format_tools(tools),
        question=question
    )

    while True:
        response = llm.generate(prompt)

        if "Answer:" in response:
            return extract_answer(response)

        action = parse_action(response)
        observation = execute_tool(action, tools)
        prompt += f"\nObservation: {observation}\n"
```

### 6. Toolformer

**Key Idea**: Model learns when to use tools

**Training**:
1. Generate tool calls with few-shot prompting
2. Execute tools, get results
3. Keep calls that improve perplexity
4. Fine-tune on augmented data

**Example**:
```
Input:  "The population of Paris is"
Output: "The population of Paris is [QA("population of Paris")] 2.1 million"
```

### 7. Agent Architectures

**Single Agent**
```
User → LLM → Tools → Response
```

**Multi-Agent**
```
User → Orchestrator → [Researcher, Writer, Coder] → Response
```

**Hierarchical**
```
User → Planner → Executors → Tools → Response
```

**Challenges**:
- Error propagation
- Context management
- Infinite loops
- Security (tool access)

### 8. Memory Systems

**Short-term**: Conversation context
```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]
```

**Long-term**: Retrieved from storage
```python
def retrieve_memories(query, k=5):
    relevant = memory_store.search(query, k=k)
    return format_memories(relevant)
```

**Working Memory**: Current task state
```python
scratchpad = {
    "goal": "Book flight to Paris",
    "steps_completed": ["Found flights", "Selected option"],
    "current_step": "Enter payment",
}
```

## Key References

### Required Reading

1. **Touvron et al. (2023) - "LLaMA: Open and Efficient Foundation Language Models"**
   - LLaMA architecture
   - [Paper](https://arxiv.org/abs/2302.13971)

2. **Yao et al. (2023) - "ReAct: Synergizing Reasoning and Acting in Language Models"**
   - ReAct framework
   - [Paper](https://arxiv.org/abs/2210.03629)

3. **Schick et al. (2023) - "Toolformer: Language Models Can Teach Themselves to Use Tools"**
   - Self-supervised tool learning
   - [Paper](https://arxiv.org/abs/2302.04761)

### Recommended Reading

4. **Su et al. (2021) - "RoFormer: Enhanced Transformer with Rotary Position Embedding"**
   - RoPE
   - [Paper](https://arxiv.org/abs/2104.09864)

5. **Dao et al. (2022) - "FlashAttention: Fast and Memory-Efficient Exact Attention"**
   - FlashAttention
   - [Paper](https://arxiv.org/abs/2205.14135)

6. **Shazeer et al. (2017) - "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"**
   - MoE foundations
   - [Paper](https://arxiv.org/abs/1701.06538)

## Practical Exercise

### Exercise 1: Long Context Handling

**Objective**: Experiment with RoPE and context extension

```python
from transformers import AutoModelForCausalLM

# Compare models with different context lengths
models = [
    "meta-llama/Llama-2-7b-hf",      # 4K context
    "mistralai/Mistral-7B-v0.1",      # 8K sliding window
]

# Test on long document QA
def test_long_context(model, document, questions):
    # Measure accuracy at different context positions
    pass
```

### Exercise 2: Build a ReAct Agent

**Objective**: Implement ReAct with multiple tools

```python
class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

tools = [
    Tool("calculator", "Evaluate math expressions", lambda expr: eval(expr)),
    Tool("search", "Search the web", web_search),
    Tool("weather", "Get weather for location", get_weather),
]

class ReActAgent:
    def __init__(self, llm, tools, max_steps=10):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, query):
        # Implement ReAct loop
        pass
```

**Test Cases**:
1. "What's 15% tip on a $67 bill?"
2. "Is it warmer in Paris or London right now?"
3. "Find recent news about AI and summarize the main points"

### Exercise 3: Compare Agent Architectures

**Objective**: Build and compare single vs multi-agent systems

| Architecture | Task Success | Latency | Token Usage |
|--------------|--------------|---------|-------------|
| Single Agent | | | |
| ReAct | | | |
| Multi-Agent | | | |

### Exercise Files
- [exercises/rope_implementation.py](./exercises/rope_implementation.py)
- [exercises/react_agent.py](./exercises/react_agent.py)
- [exercises/multi_agent.py](./exercises/multi_agent.py)

## Study Questions

1. How does MoE enable larger models without proportional compute?
2. Why does RoPE extrapolate better than absolute position embeddings?
3. What are the failure modes of ReAct agents?
4. When would you use multi-agent vs single-agent systems?
5. How do you prevent agents from taking harmful actions?

## Additional Resources

### Libraries
- [LangChain](https://python.langchain.com/) - Agent framework
- [LlamaIndex](https://www.llamaindex.ai/) - Data framework for LLMs
- [AutoGen](https://microsoft.github.io/autogen/) - Multi-agent framework

### Tutorials
- [LangChain Agent Tutorial](https://python.langchain.com/docs/modules/agents/)
- [Building Agents with Function Calling](https://platform.openai.com/docs/guides/function-calling)

### Models
- LLaMA 3 - Meta
- Mistral/Mixtral - Mistral AI
- GPT-4 - OpenAI
- Claude - Anthropic

## Next Week Preview

Week 13 covers complex reasoning and linguistics:
- Chain-of-thought prompting
- Compositional generalization
- Logical reasoning

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 13, ensure you can:
- [ ] Explain modern LLM architectural differences
- [ ] Implement RoPE positional encoding
- [ ] Build a ReAct agent with tools
- [ ] Handle long context limitations
- [ ] Design agent memory systems
