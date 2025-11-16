#!/usr/bin/env python3
"""
Generate All Course Datasets
=============================
This script generates synthetic datasets for the AI & LLM course.

Usage:
    python generate_datasets.py --all
    python generate_datasets.py --module 2
    python generate_datasets.py --house-prices --samples 10000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json


def generate_house_prices(output_dir, n_samples=5000):
    """Generate realistic house prices dataset for regression."""
    print(f"\n{'='*60}")
    print(f"Generating House Prices Dataset ({n_samples:,} samples)")
    print(f"{'='*60}")

    np.random.seed(42)

    # Generate features
    data = {
        'SquareFeet': np.random.randint(800, 4000, n_samples),
        'Bedrooms': np.random.randint(1, 6, n_samples),
        'Bathrooms': np.random.randint(1, 4, n_samples),
        'Age': np.random.randint(0, 50, n_samples),
        'LotSize': np.random.randint(2000, 10000, n_samples),
        'Garage': np.random.randint(0, 3, n_samples),
        'HasPool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Neighborhood': np.random.choice(['A', 'B', 'C'], n_samples),
        'Condition': np.random.randint(1, 6, n_samples),
        'YearBuilt': np.random.randint(1960, 2024, n_samples),
    }

    # Generate realistic prices with dependencies
    base_price = 100000
    price = (
        data['SquareFeet'] * 150 +
        data['Bedrooms'] * 20000 +
        data['Bathrooms'] * 15000 -
        data['Age'] * 2000 +
        data['LotSize'] * 10 +
        data['Garage'] * 25000 +
        data['HasPool'] * 30000 +
        (data['Condition'] * 10000) +
        np.random.randn(n_samples) * 30000 +
        base_price
    )

    # Neighborhood effect
    neighborhood_effect = {
        'A': 50000,
        'B': 0,
        'C': -30000
    }
    price += np.array([neighborhood_effect[n] for n in data['Neighborhood']])

    data['Price'] = np.maximum(price, 50000).astype(int)

    df = pd.DataFrame(data)

    # Save
    output_path = Path(output_dir) / 'house_prices.csv'
    df.to_csv(output_path, index=False)

    # Statistics
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Price range: ${df['Price'].min():,} - ${df['Price'].max():,}")
    print(f"  Mean price: ${df['Price'].mean():,.0f}")
    print(f"  Median price: ${df['Price'].median():,.0f}")

    # Correlation check
    print(f"\n  Top 3 correlated features with Price:")
    correlations = df.corr()['Price'].sort_values(ascending=False)[1:4]
    for feat, corr in correlations.items():
        print(f"    {feat:15s}: {corr:+.3f}")

    return df


def generate_credit_fraud(output_dir, n_samples=10000):
    """Generate imbalanced credit card fraud dataset."""
    print(f"\n{'='*60}")
    print(f"Generating Credit Fraud Dataset ({n_samples:,} samples)")
    print(f"{'='*60}")

    np.random.seed(42)

    # Imbalanced split: 98% normal, 2% fraud
    n_normal = int(n_samples * 0.98)
    n_fraud = n_samples - n_normal

    # Generate features (anonymized PCA components)
    n_features = 30

    # Normal transactions
    X_normal = np.random.randn(n_normal, n_features) * np.random.uniform(0.5, 2.0, n_features)

    # Fraud transactions (shifted distribution)
    X_fraud = np.random.randn(n_fraud, n_features) * np.random.uniform(0.5, 2.0, n_features)
    X_fraud += np.random.uniform(-2, 2, n_features)  # Shift some features

    # Add transaction amounts
    amounts_normal = np.random.lognormal(4, 1.5, n_normal).clip(1, 10000)
    amounts_fraud = np.random.lognormal(5, 2.0, n_fraud).clip(100, 20000)

    # Combine
    X = np.vstack([X_normal, X_fraud])
    amounts = np.hstack([amounts_normal, amounts_fraud])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    amounts = amounts[indices]
    y = y[indices]

    # Create DataFrame
    columns = [f'V{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, 'Time', np.arange(len(df)))
    df.insert(len(df.columns), 'Amount', amounts)
    df['Class'] = y.astype(int)

    # Save
    output_path = Path(output_dir) / 'credit_fraud.csv'
    df.to_csv(output_path, index=False)

    # Statistics
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution:")
    print(f"    Class 0 (Normal): {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
    print(f"    Class 1 (Fraud):  {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
    print(f"  Amount range: ${amounts.min():.2f} - ${amounts.max():.2f}")

    return df


def generate_customers(output_dir, n_samples=2000):
    """Generate customer segmentation dataset."""
    print(f"\n{'='*60}")
    print(f"Generating Customer Segmentation Dataset ({n_samples:,} samples)")
    print(f"{'='*60}")

    try:
        from sklearn.datasets import make_blobs
    except ImportError:
        print("  ⚠ scikit-learn not installed. Skipping.")
        return None

    np.random.seed(42)

    # Generate 4 customer segments
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=5,
        centers=4,
        cluster_std=1.5,
        random_state=42
    )

    # Transform to realistic customer features
    df = pd.DataFrame({
        'CustomerID': range(1, n_samples + 1),
        'Age': (X[:, 0] * 10 + 40).clip(18, 80).astype(int),
        'AnnualIncome': (X[:, 1] * 20000 + 60000).clip(20000, 200000).astype(int),
        'SpendingScore': (X[:, 2] * 20 + 50).clip(0, 100).astype(int),
        'YearsAsCustomer': (X[:, 3] * 2 + 5).clip(0, 20).round(1),
        'PurchaseFrequency': (X[:, 4] * 5 + 15).clip(0, 50).astype(int),
    })

    # Save
    output_path = Path(output_dir) / 'customers.csv'
    df.to_csv(output_path, index=False)

    # Statistics
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Age range: {df['Age'].min()} - {df['Age'].max()} years")
    print(f"  Income range: ${df['AnnualIncome'].min():,} - ${df['AnnualIncome'].max():,}")
    print(f"  Spending score: {df['SpendingScore'].min()} - {df['SpendingScore'].max()}")
    print(f"  True clusters: {len(np.unique(y_true))}")

    return df


def generate_instruction_dataset(output_dir, n_samples=1000):
    """Generate simple instruction dataset for fine-tuning."""
    print(f"\n{'='*60}")
    print(f"Generating Instruction Dataset ({n_samples:,} samples)")
    print(f"{'='*60}")

    np.random.seed(42)

    # Template-based generation
    templates = [
        {
            'instruction': 'Explain {topic} in simple terms.',
            'output': 'I\'ll explain {topic}. {explanation} This is a fundamental concept that {importance}.'
        },
        {
            'instruction': 'Write a Python function to {task}.',
            'output': 'Here\'s a Python function to {task}:\\n\\n```python\\n{code}\\n```\\n\\nThis function {description}.'
        },
        {
            'instruction': 'What is the difference between {concept1} and {concept2}?',
            'output': 'The main difference between {concept1} and {concept2} is: {difference}. {additional_info}'
        },
    ]

    # Sample topics/tasks
    topics = ['machine learning', 'neural networks', 'gradient descent', 'overfitting']
    tasks = ['sort a list', 'find prime numbers', 'reverse a string', 'calculate factorial']
    concepts = [
        ('supervised learning', 'unsupervised learning'),
        ('classification', 'regression'),
        ('training', 'testing'),
    ]

    instructions = []

    for i in range(n_samples):
        template = templates[i % len(templates)]

        if '{topic}' in template['instruction']:
            topic = topics[i % len(topics)]
            inst = template['instruction'].format(topic=topic)
            out = template['output'].format(
                topic=topic,
                explanation=f'This is explanation {i}',
                importance='helps understand AI'
            )
        elif '{task}' in template['instruction']:
            task = tasks[i % len(tasks)]
            inst = template['instruction'].format(task=task)
            out = template['output'].format(
                task=task,
                code=f'def function_{i}():\\n    pass',
                description=f'performs {task}'
            )
        else:
            c1, c2 = concepts[i % len(concepts)]
            inst = template['instruction'].format(concept1=c1, concept2=c2)
            out = template['output'].format(
                concept1=c1,
                concept2=c2,
                difference=f'Difference {i}',
                additional_info='More details here.'
            )

        instructions.append({
            'instruction': inst,
            'input': '',
            'output': out
        })

    # Save
    output_path = Path(output_dir) / 'instructions.json'
    with open(output_path, 'w') as f:
        json.dump(instructions, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Samples: {len(instructions):,}")
    print(f"  Format: Alpaca-style instruction dataset")

    return instructions


def main():
    parser = argparse.ArgumentParser(
        description='Generate datasets for AI & LLM course'
    )
    parser.add_argument('--all', action='store_true',
                       help='Generate all datasets')
    parser.add_argument('--module', type=int,
                       help='Generate datasets for specific module')
    parser.add_argument('--house-prices', action='store_true',
                       help='Generate house prices dataset')
    parser.add_argument('--credit-fraud', action='store_true',
                       help='Generate credit fraud dataset')
    parser.add_argument('--customers', action='store_true',
                       help='Generate customer segmentation dataset')
    parser.add_argument('--instructions', action='store_true',
                       help='Generate instruction dataset')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of samples (default: 5000)')
    parser.add_argument('--output', default='./module_data',
                       help='Output directory (default: ./module_data)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("AI & LLM Course Dataset Generator")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")

    # Generate datasets
    if args.all or args.module == 2 or args.house_prices:
        generate_house_prices(output_dir, args.samples)

    if args.all or args.module == 2 or args.credit_fraud:
        generate_credit_fraud(output_dir, args.samples)

    if args.all or args.module == 2 or args.customers:
        generate_customers(output_dir, min(args.samples, 2000))

    if args.all or args.module == 5 or args.instructions:
        generate_instruction_dataset(output_dir, args.samples)

    print("\n" + "="*60)
    print("✓ Dataset generation complete!")
    print("="*60)
    print(f"\nAll files saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Verify the generated datasets")
    print("  2. Use them in course examples")
    print("  3. Experiment with different parameters")


if __name__ == '__main__':
    main()
