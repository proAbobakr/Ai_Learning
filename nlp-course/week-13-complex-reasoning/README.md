# Week 13: Complex Reasoning & Linguistics

## Overview

This week explores how language models handle complex reasoning tasks and what linguistic capabilities they possess. We'll cover chain-of-thought prompting, compositional generalization, and the relationship between language and thought.

## Learning Objectives

By the end of this week, you will be able to:
- Apply chain-of-thought prompting effectively
- Design prompts for multi-step reasoning
- Understand compositional generalization challenges
- Evaluate models on linguistic structure tasks
- Analyze reasoning failures in LLMs
- Apply techniques for improved reasoning

## Core Topics

### 1. Chain-of-Thought Prompting

**Key Insight**: Let models "think step by step"

**Standard Prompting**:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: 11
```

**Chain-of-Thought**:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger starts with 5 balls. He buys 2 cans with 3 balls each.
   2 cans × 3 balls = 6 balls. 5 + 6 = 11 balls.
```

**Why It Works**:
- Breaks complex problems into steps
- Allocates more compute to harder problems
- Makes reasoning transparent
- Reduces errors in intermediate steps

### 2. Zero-Shot Chain-of-Thought

**The Magic Phrase**: "Let's think step by step"

```python
def zero_shot_cot(question):
    prompt = f"{question}\n\nLet's think step by step."
    reasoning = model.generate(prompt)
    # Extract final answer
    return extract_answer(reasoning)
```

**Kojima et al. (2022)**: Just adding this phrase improves accuracy dramatically on reasoning tasks.

**Variations**:
- "Let's work this out in a step by step way to be sure we have the right answer."
- "Think through this carefully."
- "Before answering, reason through the problem."

### 3. Few-Shot Chain-of-Thought

**Provide Examples with Reasoning**:
```
Q: There are 15 trees in the grove. Grove workers will plant trees in the
   grove today. After they are done, there will be 21 trees. How many trees
   did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more
   were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many
   cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
   The answer is 5.

Q: [New question here]
A:
```

### 4. Self-Consistency

**Idea**: Sample multiple reasoning paths, vote on answer

```python
def self_consistency(question, n_samples=5, temperature=0.7):
    answers = []

    for _ in range(n_samples):
        response = model.generate(
            f"{question}\nLet's think step by step.",
            temperature=temperature
        )
        answer = extract_answer(response)
        answers.append(answer)

    # Majority vote
    return Counter(answers).most_common(1)[0][0]
```

**Benefits**:
- More robust than single sample
- Different paths catch different errors
- Particularly effective on math/logic

### 5. Tree of Thoughts

**Extension of CoT**: Explore multiple reasoning branches

```
                    [Problem]
                       │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
       [Thought1] [Thought2] [Thought3]
           │          │          │
         eval       eval       eval
           │          ×          │
       [Thought4]           [Thought5]
           │                    │
         eval                 eval
           │                    ×
       [Solution]
```

**Implementation**:
```python
def tree_of_thoughts(problem, breadth=3, depth=3):
    def evaluate_thought(thought):
        prompt = f"Rate this reasoning (1-10):\n{thought}"
        return model.score(prompt)

    def generate_thoughts(state, k):
        prompt = f"Given:\n{state}\nPropose {k} different next steps:"
        return model.generate(prompt, n=k)

    queue = [problem]
    for level in range(depth):
        candidates = []
        for state in queue:
            thoughts = generate_thoughts(state, breadth)
            for thought in thoughts:
                score = evaluate_thought(state + thought)
                candidates.append((state + thought, score))

        # Keep top thoughts
        queue = [c[0] for c in sorted(candidates, key=lambda x: -x[1])[:breadth]]

    return queue[0]
```

### 6. Compositional Generalization

**Key Question**: Can models understand novel combinations of known parts?

**Example (SCAN dataset)**:
```
Training:
  "jump" → JUMP
  "walk twice" → WALK WALK
  "jump right" → TURN_RIGHT JUMP

Test:
  "jump twice" → JUMP JUMP  ✓ (seen pattern)
  "jump around right" → ?   (novel combination)
```

**COGS Dataset** (Kim & Linzen, 2020):
- Tests systematic generalization in semantic parsing
- Novel combinations of seen primitives
- Models often fail despite understanding parts

**Why Models Struggle**:
- Memorize patterns, not rules
- Distribution shift from training
- Lack of explicit compositional structure

### 7. Linguistic Structure

**What Do LLMs Learn About Language?**

**Syntax**:
- Subject-verb agreement: "The keys to the cabinet *are*..."
- Long-distance dependencies
- Garden path sentences

**Semantics**:
- Negation: "I don't not like it"
- Quantifiers: "All circles are blue" vs "Some circles are blue"
- Presuppositions

**Pragmatics**:
- Implicature: "Can you pass the salt?" (request, not question)
- Context sensitivity

**Harris's Distributional Hypothesis** (1954):
> "Words that occur in similar contexts tend to have similar meanings."

### 8. Logical Reasoning

**Types of Reasoning**:

**Deductive**:
```
All men are mortal.
Socrates is a man.
Therefore: Socrates is mortal.
```

**Inductive**:
```
The sun rose today.
The sun rose yesterday.
The sun rose the day before.
Therefore: The sun will rise tomorrow.
```

**Abductive**:
```
The grass is wet.
Best explanation: It rained.
```

**LLM Performance**:
- Deductive: Generally good with prompting
- Inductive: Pattern recognition works
- Abductive: Struggles with novel explanations

### 9. Reasoning Failures

**Common Issues**:

1. **Reversal Curse**:
   - Model knows "A is B" but not "B is A"
   - "Tom Cruise's mother is Mary Lee" ✓
   - "Who is Mary Lee's son?" ✗

2. **Negation**:
   - "Write a sentence without the word 'the'" → uses "the"

3. **Math Reasoning**:
   - Inconsistent on multi-step problems
   - Sensitive to problem phrasing

4. **Logical Fallacies**:
   - Affirming the consequent
   - False dilemma

### 10. Improving Reasoning

**Techniques**:

1. **Scratchpad Training**
   - Train on intermediate steps
   - Model learns to show work

2. **Verification**
   - Generate answer
   - Ask model to verify
   - Iterate

3. **Process Reward Models**
   - Reward intermediate steps, not just final answer
   - Encourages valid reasoning paths

4. **Symbolic Integration**
   - Use code/calculators for computation
   - LLM for planning, tools for execution

## Key References

### Required Reading

1. **Wei et al. (2022) - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"**
   - Original CoT paper
   - [Paper](https://arxiv.org/abs/2201.11903)

2. **Kojima et al. (2022) - "Large Language Models are Zero-Shot Reasoners"**
   - "Let's think step by step"
   - [Paper](https://arxiv.org/abs/2205.11916)

3. **Kim & Linzen (2020) - "COGS: A Compositional Generalization Challenge"**
   - Compositional generalization benchmark
   - [Paper](https://arxiv.org/abs/2010.05465)

### Recommended Reading

4. **Wang et al. (2023) - "Self-Consistency Improves Chain of Thought Reasoning"**
   - Self-consistency decoding
   - [Paper](https://arxiv.org/abs/2203.11171)

5. **Yao et al. (2023) - "Tree of Thoughts"**
   - Deliberate reasoning with search
   - [Paper](https://arxiv.org/abs/2305.10601)

6. **Harris (1954) - "Distributional Structure"**
   - Classic paper on distributional semantics
   - [Paper](https://www.tandfonline.com/doi/abs/10.1080/00437956.1954.11659520)

## Practical Exercise

### Exercise 1: Chain-of-Thought Implementation

**Objective**: Compare prompting strategies on reasoning tasks

```python
# Test on GSM8K (grade school math) problems
def test_prompting_strategies(problem):
    strategies = {
        "direct": direct_prompt(problem),
        "zero_shot_cot": zero_shot_cot(problem),
        "few_shot_cot": few_shot_cot(problem),
        "self_consistency": self_consistency(problem, n=5),
    }

    return strategies

# Measure accuracy across strategies
```

### Exercise 2: Compositional Generalization

**Objective**: Evaluate model on COGS-style tasks

```python
# Create simple compositional task
training_data = [
    ("red circle", "DRAW red SHAPE circle"),
    ("blue square", "DRAW blue SHAPE square"),
    ("big red circle", "DRAW big red SHAPE circle"),
]

test_data = [
    ("big blue square", "DRAW big blue SHAPE square"),  # Novel combination
]

def evaluate_composition(model, train, test):
    # Fine-tune or few-shot on training
    # Evaluate on test combinations
    pass
```

### Exercise 3: Logical Reasoning Evaluation

**Objective**: Test logical reasoning capabilities

```python
# Syllogism test cases
syllogisms = [
    {
        "premises": ["All A are B", "All B are C"],
        "conclusion": "All A are C",
        "valid": True
    },
    {
        "premises": ["Some A are B", "All B are C"],
        "conclusion": "Some A are C",
        "valid": True
    },
    {
        "premises": ["All A are B", "Some B are C"],
        "conclusion": "Some A are C",
        "valid": False  # Invalid!
    }
]

def test_logical_reasoning(model, syllogisms):
    for s in syllogisms:
        prompt = f"""Given:
{chr(10).join(s['premises'])}

Is the following conclusion valid? {s['conclusion']}
Think step by step."""

        response = model.generate(prompt)
        # Evaluate correctness
```

### Exercise Files
- [exercises/cot_prompting.py](./exercises/cot_prompting.py)
- [exercises/compositional_test.py](./exercises/compositional_test.py)
- [exercises/logical_reasoning.py](./exercises/logical_reasoning.py)

## Study Questions

1. Why does chain-of-thought improve reasoning performance?
2. What's the difference between memorization and compositional understanding?
3. Why do models struggle with negation?
4. How does self-consistency improve over single-sample decoding?
5. What are the limits of prompting-based reasoning improvements?

## Additional Resources

### Benchmarks
- GSM8K - Grade school math
- MATH - Competition mathematics
- MMLU - Multitask language understanding
- BIG-Bench - Beyond the Imitation Game

### Tutorials
- [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)
- [Prompt Engineering Guide - CoT](https://www.promptingguide.ai/techniques/cot)

### Papers
- Scratchpad paper (Nye et al., 2021)
- Least-to-Most prompting (Zhou et al., 2022)

## Next Week Preview

Week 14 is the final week, covering:
- Multilingual NLP
- Cross-lingual transfer
- Course wrap-up and final project

---

**Estimated Time**: 15-20 hours

**Checkpoint**: Before moving to Week 14, ensure you can:
- [ ] Apply chain-of-thought prompting
- [ ] Implement self-consistency decoding
- [ ] Evaluate compositional generalization
- [ ] Analyze reasoning failures
- [ ] Design prompts for complex reasoning tasks
