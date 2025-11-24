# Week 8: Experimental Design & Human Annotation

## Overview

This week shifts focus to the scientific methodology of NLP research. Good experimental design and high-quality data annotation are crucial for valid conclusions. We'll cover best practices for dataset creation, evaluation, and reporting.

## Learning Objectives

By the end of this week, you will be able to:
- Design rigorous NLP experiments
- Identify and avoid common pitfalls in ML evaluation
- Create annotation guidelines for NLP tasks
- Measure inter-annotator agreement (Cohen's Kappa, Krippendorff's Alpha)
- Apply best practices for data collection
- Write clear data statements and model cards

## Core Topics

### 1. Principles of Experimental Design

**The Scientific Method in NLP**
1. **Hypothesis**: Clear, testable claim
2. **Variables**: Independent (what you change), dependent (what you measure)
3. **Controls**: Baseline comparisons
4. **Replication**: Multiple runs, random seeds

**Common Questions to Answer**:
- Does method A outperform method B?
- What's the effect of hyperparameter X?
- How does the model behave on edge cases?

### 2. Dataset Splits and Evaluation

**Standard Splits**
```
Dataset
├── Training (70-80%)    ← Model learns from this
├── Validation (10-15%)  ← Hyperparameter tuning
└── Test (10-15%)        ← Final evaluation ONCE
```

**Critical Rules**:
- Never tune on test set
- Report test results only at the end
- Use validation for all development decisions

**Cross-Validation**
```
Fold 1: [Val] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Val] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Val] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Val] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Val]
```
- Useful for small datasets
- More robust estimates
- Computationally expensive

### 3. Common Pitfalls in ML/NLP

**Data Leakage**
- Training data appears in test set
- Features computed using test information
- Temporal leakage (future data predicting past)

**Overfitting to Test Set**
- Running many experiments on test set
- Selecting best result from multiple runs
- Solution: single final evaluation

**Cherry-Picking Results**
- Only reporting best random seed
- Selectively showing examples
- Solution: report mean ± std over seeds

**Inadequate Baselines**
- Comparing to weak baselines
- Not comparing to simple methods
- Always include: random, majority class, simple model

**Dataset Artifacts**
- Spurious correlations in data
- Annotation artifacts (e.g., negation words → contradiction)
- Solution: adversarial evaluation, stress tests

### 4. Statistical Significance

**Why It Matters**
- Small improvements might be noise
- Need to distinguish real effects from random variation

**Bootstrap Testing**
```python
def bootstrap_test(scores_a, scores_b, n_bootstrap=10000):
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    combined = np.concatenate([scores_a, scores_b])

    count = 0
    for _ in range(n_bootstrap):
        np.random.shuffle(combined)
        new_a = combined[:len(scores_a)]
        new_b = combined[len(scores_a):]
        if np.mean(new_a) - np.mean(new_b) >= observed_diff:
            count += 1

    p_value = count / n_bootstrap
    return p_value
```

**Reporting**:
- Report p-values for key comparisons
- Use confidence intervals
- Multiple comparison correction if many tests

### 5. Human Annotation Basics

**When to Use Human Annotation**
- Creating new datasets
- Evaluating subjective qualities
- Gold standard for automatic metrics

**Annotation Pipeline**
```
Task Definition → Guidelines → Pilot → Train Annotators → Annotate → Adjudicate
```

### 6. Writing Annotation Guidelines

**Good Guidelines Include**:
1. **Task description**: What are annotators doing?
2. **Label definitions**: Clear, mutually exclusive categories
3. **Edge cases**: How to handle ambiguous examples
4. **Examples**: Annotated samples for each category
5. **Process**: How to use the annotation tool

**Example**: Sentiment Annotation Guidelines
```markdown
## Task
Label each tweet with its sentiment: POSITIVE, NEGATIVE, or NEUTRAL.

## Definitions
- POSITIVE: Expresses satisfaction, happiness, approval, or enthusiasm
- NEGATIVE: Expresses dissatisfaction, anger, sadness, or criticism
- NEUTRAL: Factual statements, questions, or mixed sentiment

## Edge Cases
- Sarcasm: Label based on true intent (often NEGATIVE)
- Questions: NEUTRAL unless clearly rhetorical
- Mixed: If equally positive and negative, use NEUTRAL

## Examples
"I love this new feature!" → POSITIVE
"This update broke everything." → NEGATIVE
"The update was released today." → NEUTRAL
"Oh great, another bug." (sarcastic) → NEGATIVE
```

### 7. Inter-Annotator Agreement

**Why Measure Agreement?**
- Validates task clarity
- Identifies problematic examples
- Estimates human performance ceiling

**Cohen's Kappa (2 annotators)**
```
κ = (p_o - p_e) / (1 - p_e)

p_o = observed agreement
p_e = expected agreement by chance
```

**Interpretation**:
- κ < 0.20: Slight
- 0.21-0.40: Fair
- 0.41-0.60: Moderate
- 0.61-0.80: Substantial
- 0.81-1.00: Almost perfect

```python
from sklearn.metrics import cohen_kappa_score

annotator1 = [1, 1, 0, 1, 0, 0, 1]
annotator2 = [1, 0, 0, 1, 0, 1, 1]

kappa = cohen_kappa_score(annotator1, annotator2)
print(f"Cohen's Kappa: {kappa:.3f}")
```

**Krippendorff's Alpha (multiple annotators, missing data)**
- More flexible than Kappa
- Handles multiple annotators
- Works with different data types (nominal, ordinal, interval)

```python
import krippendorff

# Each row is an annotator, each column is an item
# Use np.nan for missing annotations
annotations = [
    [1, 2, 3, 3, 2, 1, np.nan],
    [1, 2, 3, 3, 2, 2, 2],
    [np.nan, 2, 3, 3, 2, 1, 2],
]

alpha = krippendorff.alpha(annotations, level_of_measurement='nominal')
print(f"Krippendorff's Alpha: {alpha:.3f}")
```

### 8. Data Statements and Documentation

**Data Statements** (Bender & Friedman, 2018)
Document:
- Curation rationale
- Language variety
- Speaker demographics
- Annotator demographics
- Speech situation
- Text characteristics
- Provenance

**Model Cards** (Mitchell et al., 2019)
- Model details (architecture, training data)
- Intended use and users
- Limitations and biases
- Evaluation results across groups
- Ethical considerations

## Key References

### Required Reading

1. **Bender & Friedman (2018) - "Data Statements for Natural Language Processing"**
   - Framework for documenting datasets
   - [Paper](https://aclanthology.org/Q18-1041/)

2. **Lones (2021) - "How to Avoid Machine Learning Pitfalls"**
   - Comprehensive guide to ML best practices
   - [Paper](https://arxiv.org/abs/2108.02497)

### Recommended Reading

3. **Artstein & Poesio (2008) - "Inter-Coder Agreement for Computational Linguistics"**
   - Detailed treatment of agreement metrics
   - [Paper](https://aclanthology.org/J08-4004/)

4. **Mitchell et al. (2019) - "Model Cards for Model Reporting"**
   - Framework for model documentation
   - [Paper](https://arxiv.org/abs/1810.03993)

5. **Gururangan et al. (2018) - "Annotation Artifacts in Natural Language Inference Data"**
   - Dataset artifacts analysis
   - [Paper](https://aclanthology.org/N18-2017/)

## Practical Exercise

### Exercise 1: Dataset Collection & Annotation

**Task**: Create a small sentiment dataset with multiple annotators

**Steps**:
1. Collect 100 tweets/reviews on a topic
2. Write annotation guidelines
3. Have 2-3 people annotate independently
4. Calculate inter-annotator agreement
5. Adjudicate disagreements

**Deliverables**:
- Annotation guidelines document
- Annotated dataset
- Agreement metrics (Kappa/Alpha)
- Analysis of disagreements

### Exercise 2: Experiment Design Audit

**Task**: Review a published paper's methodology

**Questions to Answer**:
1. How was the test set used?
2. Are baselines appropriate?
3. Is statistical significance reported?
4. Could there be data leakage?
5. Are random seeds reported?

### Exercise 3: Write a Data Statement

**Task**: Document an existing NLP dataset

**Template**:
```markdown
## Data Statement for [Dataset Name]

### Curation Rationale
[Why was this dataset created?]

### Language Variety
[What language(s)? What dialect(s)?]

### Speaker Demographics
[Who produced the text?]

### Annotator Demographics
[Who labeled the data?]

### Collection Process
[How was data gathered?]

### Limitations
[What should users be aware of?]
```

### Exercise Files
- [exercises/annotation_guidelines_template.md](./exercises/annotation_guidelines_template.md)
- [exercises/agreement_calculator.py](./exercises/agreement_calculator.py)
- [exercises/experiment_checklist.md](./exercises/experiment_checklist.md)

## Study Questions

1. Why is it problematic to tune hyperparameters on the test set?
2. How does Cohen's Kappa account for chance agreement?
3. What's the difference between accuracy and inter-annotator agreement?
4. Why might low agreement indicate a problem with guidelines vs annotators?
5. How can dataset artifacts lead to misleading model performance?

## Additional Resources

### Tools
- [Label Studio](https://labelstud.io/) - Open-source annotation tool
- [Prodigy](https://prodi.gy/) - Commercial annotation tool
- [Doccano](https://github.com/doccano/doccano) - Text annotation

### Guidelines
- [ACL Responsible NLP Checklist](https://aclrollingreview.org/responsibleNLPresearch/)
- [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)

### Tutorials
- [Annotation Best Practices](https://explosion.ai/blog/prodigy-annotation-tool-active-learning)

## Next Week Preview

Week 9 introduces retrieval systems and RAG:
- Classic information retrieval (BM25)
- Dense Passage Retrieval
- Retrieval-augmented generation

---

**Estimated Time**: 12-15 hours

**Checkpoint**: Before moving to Week 9, ensure you can:
- [ ] Design an experiment with proper train/val/test splits
- [ ] Identify common ML pitfalls in paper methodology
- [ ] Write clear annotation guidelines
- [ ] Calculate and interpret Cohen's Kappa
- [ ] Write a data statement for a dataset
