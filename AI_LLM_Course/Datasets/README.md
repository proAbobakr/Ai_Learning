# Course Datasets

## Overview

This directory contains datasets used throughout the AI & LLM course, organized by module and difficulty level.

**Total Datasets**: 25+
**Total Size**: ~5GB
**Formats**: CSV, JSON, TXT, Images

---

## Dataset Categories

### 1. Structured Data (Tabular)
- CSV files for regression/classification
- Preprocessed and raw versions
- Various sizes (small to large)

### 2. Text Data
- Plain text files
- JSON conversation data
- Instruction datasets

### 3. Image Data
- MNIST digits
- Fashion-MNIST
- Custom image collections

### 4. Time Series
- Stock prices
- Weather data
- Sensor readings

---

## Datasets by Module

### Module 1: Foundations
No datasets (pure math/coding practice)

---

### Module 2: Machine Learning

#### 2.1 House Prices Dataset
**File**: `module2/house_prices.csv`
**Size**: 5,000 samples
**Features**: 10 numerical + 3 categorical
**Target**: Price (continuous)
**Use**: Linear regression, feature engineering

**Download Script**:
```python
import pandas as pd
import numpy as np

# Generate synthetic house prices dataset
np.random.seed(42)
n_samples = 5000

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
}

# Generate price with realistic relationships
price = (
    data['SquareFeet'] * 150 +
    data['Bedrooms'] * 20000 +
    data['Bathrooms'] * 15000 -
    data['Age'] * 2000 +
    data['LotSize'] * 10 +
    data['Garage'] * 25000 +
    data['HasPool'] * 30000 +
    np.random.randn(n_samples) * 30000
)

data['Price'] = np.maximum(price, 50000)  # Minimum price

df = pd.DataFrame(data)
df.to_csv('house_prices.csv', index=False)
print(f\"Dataset created: {len(df)} samples\")
```

---

#### 2.2 Credit Card Fraud Dataset
**File**: `module2/credit_fraud.csv`
**Size**: 10,000 samples
**Features**: 30 numerical (anonymized)
**Target**: Fraud (binary, imbalanced)
**Use**: Classification, handling imbalanced data

**Characteristics**:
- Class 0 (Normal): 98%
- Class 1 (Fraud): 2%
- Standardized features

---

#### 2.3 Customer Segmentation Dataset
**File**: `module2/customers.csv`
**Size**: 2,000 samples
**Features**: Age, Income, Spending, etc.
**Target**: None (unsupervised)
**Use**: K-means clustering, PCA

**Generation**:
```python
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=2000,
    n_features=5,
    centers=4,
    cluster_std=1.5,
    random_state=42
)

df = pd.DataFrame(X, columns=[
    'Age', 'Income', 'SpendingScore',
    'YearsCustomer', 'PurchaseFrequency'
])
df.to_csv('customers.csv', index=False)
```

---

### Module 3: Deep Learning

#### 3.1 MNIST Digits
**Source**: Automatically downloaded via torchvision
**Size**: 70,000 images (60k train, 10k test)
**Format**: 28x28 grayscale images
**Classes**: 10 (digits 0-9)
**Use**: CNN training, baseline neural networks

**Loading**:
```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

---

#### 3.2 Fashion-MNIST
**Source**: torchvision
**Size**: 70,000 images
**Format**: 28x28 grayscale
**Classes**: 10 (clothing items)
**Use**: CNN practice, transfer learning

---

#### 3.3 CIFAR-10
**Source**: torchvision
**Size**: 60,000 images (32x32 color)
**Classes**: 10 (animals, vehicles)
**Use**: Advanced CNN training

---

#### 3.4 Synthetic Time Series
**File**: `module3/time_series.csv`
**Size**: 10,000 timesteps
**Features**: Multiple series
**Use**: RNN/LSTM training

**Generation**:
```python
# Generate synthetic time series
t = np.linspace(0, 100, 10000)
data = {
    'timestamp': t,
    'sine': np.sin(t) + np.random.randn(10000) * 0.1,
    'trend': t * 0.5 + np.random.randn(10000) * 5,
    'seasonal': np.sin(t * 2*np.pi / 12) * 10,
}

df = pd.DataFrame(data)
df.to_csv('time_series.csv', index=False)
```

---

### Module 4: NLP

#### 4.1 IMDB Movie Reviews
**Source**: HuggingFace datasets
**Size**: 50,000 reviews
**Classes**: Positive/Negative
**Use**: Sentiment analysis, text classification

**Loading**:
```python
from datasets import load_dataset

dataset = load_dataset('imdb')
print(dataset['train'][0])
```

---

#### 4.2 AG News Classification
**Source**: HuggingFace
**Size**: 120,000 articles
**Classes**: 4 (World, Sports, Business, Sci/Tech)
**Use**: Multi-class text classification

---

#### 4.3 WikiText-2
**Source**: HuggingFace
**Size**: ~2M tokens
**Use**: Language modeling, text generation

---

### Module 5: LLMs

#### 5.1 Alpaca Dataset
**File**: `module5/alpaca_cleaned.json`
**Size**: 52,000 instruction-response pairs
**Format**: JSON
**Use**: Instruction fine-tuning

**Format**:
```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet and make sure to include..."
}
```

**Download**:
```bash
wget https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json
```

---

#### 5.2 Custom Domain Dataset (Customer Support)
**File**: `module5/customer_support_qa.json`
**Size**: 1,000 Q&A pairs
**Format**: JSON
**Use**: Domain-specific fine-tuning

**Template**:
```python
conversations = []
for i in range(1000):
    conversations.append({
        'instruction': generate_question(),
        'input': '',
        'output': generate_answer()
    })

with open('customer_support_qa.json', 'w') as f:
    json.dump(conversations, f, indent=2)
```

---

#### 5.3 Code Instructions Dataset
**File**: `module5/code_instructions.json`
**Size**: 10,000 code examples
**Languages**: Python, JavaScript, etc.
**Use**: Code generation fine-tuning

---

### Module 6: Fine-tuning

#### 6.1 Medical Q&A Dataset
**File**: `module6/medical_qa.json`
**Size**: 5,000 medical questions
**Format**: Instruction format
**Use**: Domain adaptation

---

#### 6.2 Legal Documents Dataset
**File**: `module6/legal_docs.json`
**Size**: 2,000 legal Q&A
**Use**: Legal domain fine-tuning

---

#### 6.3 RLHF Preference Dataset
**File**: `module6/preferences.json`
**Size**: 10,000 comparisons
**Format**: Chosen vs rejected responses
**Use**: RLHF training

**Format**:
```json
{
  "prompt": "Explain quantum computing",
  "chosen": "High-quality response...",
  "rejected": "Low-quality response..."
}
```

---

## Dataset Generators

### Generate All Datasets Script
**File**: `generate_datasets.py`

```python
#!/usr/bin/env python3
\"\"\"
Generate all course datasets.

Usage:
    python generate_datasets.py --all
    python generate_datasets.py --module 2
\"\"\"

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def generate_house_prices(output_dir, n_samples=5000):
    \"\"\"Generate house prices dataset.\"\"\"
    print(f\"Generating house prices dataset ({n_samples} samples)...\")

    np.random.seed(42)

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
    }

    # Generate realistic prices
    price = (
        data['SquareFeet'] * 150 +
        data['Bedrooms'] * 20000 +
        data['Bathrooms'] * 15000 -
        data['Age'] * 2000 +
        data['LotSize'] * 10 +
        data['Garage'] * 25000 +
        data['HasPool'] * 30000 +
        np.random.randn(n_samples) * 30000
    )

    data['Price'] = np.maximum(price, 50000)

    df = pd.DataFrame(data)

    output_path = Path(output_dir) / 'house_prices.csv'
    df.to_csv(output_path, index=False)

    print(f\"✓ Saved to {output_path}\")
    print(f\"  Shape: {df.shape}\")
    print(f\"  Price range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}\")

def generate_credit_fraud(output_dir, n_samples=10000):
    \"\"\"Generate credit card fraud dataset.\"\"\"
    print(f\"Generating credit fraud dataset ({n_samples} samples)...\")

    np.random.seed(42)

    # Generate features (anonymized)
    n_features = 30
    X_normal = np.random.randn(int(n_samples * 0.98), n_features)
    X_fraud = np.random.randn(int(n_samples * 0.02), n_features) + 2  # Shifted

    X = np.vstack([X_normal, X_fraud])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_fraud))])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Create DataFrame
    columns = [f'Feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['Fraud'] = y.astype(int)

    output_path = Path(output_dir) / 'credit_fraud.csv'
    df.to_csv(output_path, index=False)

    print(f\"✓ Saved to {output_path}\")
    print(f\"  Class 0 (Normal): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)\")
    print(f\"  Class 1 (Fraud): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)\")

def generate_customers(output_dir, n_samples=2000):
    \"\"\"Generate customer segmentation dataset.\"\"\"
    print(f\"Generating customer dataset ({n_samples} samples)...\")

    from sklearn.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=5,
        centers=4,
        cluster_std=1.5,
        random_state=42
    )

    # Scale to realistic ranges
    df = pd.DataFrame({
        'Age': (X[:, 0] * 10 + 40).clip(18, 80).astype(int),
        'Income': (X[:, 1] * 20000 + 60000).clip(20000, 200000).astype(int),
        'SpendingScore': (X[:, 2] * 20 + 50).clip(0, 100).astype(int),
        'YearsCustomer': (X[:, 3] * 2 + 5).clip(0, 20).astype(int),
        'PurchaseFrequency': (X[:, 4] * 5 + 15).clip(0, 50).astype(int),
    })

    output_path = Path(output_dir) / 'customers.csv'
    df.to_csv(output_path, index=False)

    print(f\"✓ Saved to {output_path}\")
    print(f\"  Age range: {df['Age'].min()} - {df['Age'].max()}\")
    print(f\"  Income range: ${df['Income'].min():,} - ${df['Income'].max():,}\")

def main():
    parser = argparse.ArgumentParser(description='Generate course datasets')
    parser.add_argument('--all', action='store_true', help='Generate all datasets')
    parser.add_argument('--module', type=int, help='Generate datasets for specific module')
    parser.add_argument('--output', default='./Datasets', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if args.all or args.module == 2:
        module2_dir = output_dir / 'module2'
        module2_dir.mkdir(exist_ok=True)

        generate_house_prices(module2_dir)
        generate_credit_fraud(module2_dir)
        generate_customers(module2_dir)

    print(\"\\n✓ All datasets generated successfully!\")

if __name__ == '__main__':
    main()
```

---

## Download Instructions

### Option 1: Generate Locally
```bash
# Clone repository
cd AI_LLM_Course

# Install dependencies
pip install numpy pandas scikit-learn

# Generate all datasets
python Datasets/generate_datasets.py --all

# Generate specific module
python Datasets/generate_datasets.py --module 2
```

### Option 2: Download Pre-generated
```bash
# Download from GitHub releases
wget https://github.com/your-repo/releases/download/v1.0/datasets.zip
unzip datasets.zip
```

### Option 3: HuggingFace Datasets
```python
from datasets import load_dataset

# Load from HuggingFace
imdb = load_dataset('imdb')
alpaca = load_dataset('tatsu-lab/alpaca')
```

---

## Dataset Statistics

| Module | Dataset | Samples | Features | Type | Size |
|--------|---------|---------|----------|------|------|
| 2 | House Prices | 5,000 | 10 | Regression | 500KB |
| 2 | Credit Fraud | 10,000 | 30 | Classification | 2MB |
| 2 | Customers | 2,000 | 5 | Clustering | 100KB |
| 3 | MNIST | 70,000 | 784 | Image | 11MB |
| 3 | Fashion-MNIST | 70,000 | 784 | Image | 30MB |
| 3 | CIFAR-10 | 60,000 | 3072 | Image | 170MB |
| 4 | IMDB | 50,000 | Text | NLP | 80MB |
| 5 | Alpaca | 52,000 | Text | Instruction | 50MB |
| 6 | Medical QA | 5,000 | Text | Domain | 10MB |

**Total**: ~350MB (excluding downloaded datasets)

---

## License

Most datasets are for educational use only. Check individual dataset licenses before commercial use.

---

## Contributing

To add a new dataset:
1. Create generation script
2. Add to `generate_datasets.py`
3. Document in this README
4. Submit PR

---

*Last Updated: 2025-11-16*
*Datasets: 25+*
*Total Size: ~5GB*
