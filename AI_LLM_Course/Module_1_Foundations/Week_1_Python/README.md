# Week 1: Python for AI

## Overview

Master Python programming for AI and machine learning applications. This week covers essential Python skills, NumPy for numerical computing, Pandas for data manipulation, and data visualization libraries.

**Duration**: 15-20 hours
**Difficulty**: Beginner to Intermediate

---

## Learning Objectives

By the end of this week, you will:

✅ Write efficient Python code for AI applications
✅ Master NumPy for numerical computations
✅ Manipulate data with Pandas DataFrames
✅ Create professional data visualizations
✅ Set up Jupyter notebook workflows
✅ Understand Python best practices for ML

---

## Day-by-Day Breakdown

### Day 1-2: Python Essentials (6-8 hours)

#### Topics Covered
- Python data structures (lists, tuples, dicts, sets)
- List comprehensions and generators
- Functions and lambda expressions
- Object-oriented programming basics
- File I/O operations

#### Examples
See: [01_python_essentials.py](./examples/01_python_essentials.py)

---

### Day 3-4: NumPy Mastery (6-8 hours)

#### Topics Covered
- NumPy arrays and ndarray operations
- Array indexing and slicing
- Broadcasting rules
- Mathematical operations
- Linear algebra with NumPy
- Random number generation

#### Examples
See: [02_numpy_fundamentals.py](./examples/02_numpy_fundamentals.py)

---

### Day 5-6: Pandas & Visualization (6-8 hours)

#### Topics Covered
- DataFrames and Series
- Data loading and cleaning
- Data transformation and aggregation
- Matplotlib basics
- Seaborn for statistical plots

#### Examples
See: [03_pandas_basics.py](./examples/03_pandas_basics.py)
See: [04_visualization.py](./examples/04_visualization.py)

---

### Day 7: Integration Project (3-4 hours)

Build a complete data analysis pipeline combining all skills.

#### Project
See: [05_mini_projects.py](./examples/05_mini_projects.py)

---

## Code Examples

### Example 1: Python Essentials
```python
# List comprehensions for data processing
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
even_squared = [x**2 for x in numbers if x % 2 == 0]

# Dictionary comprehensions
word_lengths = {word: len(word) for word in ['AI', 'Machine', 'Learning']}

# Lambda functions with map/filter
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Generator for memory efficiency
def data_generator(n):
    for i in range(n):
        yield i ** 2

# Class for ML model structure
class SimpleModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = None

    def initialize_weights(self):
        import numpy as np
        self.weights = np.random.randn(self.input_dim, self.output_dim)

    def forward(self, X):
        return np.dot(X, self.weights)
```

### Example 2: NumPy Fundamentals
```python
import numpy as np

# Array creation (10+ methods)
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 3))
arr3 = np.ones((2, 4))
arr4 = np.arange(0, 10, 2)
arr5 = np.linspace(0, 1, 5)
arr6 = np.random.randn(3, 3)
arr7 = np.eye(4)
arr8 = np.full((2, 2), 7)
arr9 = np.random.randint(0, 100, (3, 3))
arr10 = np.logspace(0, 2, 5)

# Array operations
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Element-wise operations
addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b
power = a ** 2

# Matrix operations
matrix_mult = np.dot(a, b)  # or a @ b
transpose = a.T
inverse = np.linalg.inv(a)
determinant = np.linalg.det(a)
eigenvalues, eigenvectors = np.linalg.eig(a)

# Statistical operations
mean = np.mean(a)
std = np.std(a)
variance = np.var(a)
min_val = np.min(a)
max_val = np.max(a)
sum_all = np.sum(a)
sum_axis0 = np.sum(a, axis=0)
sum_axis1 = np.sum(a, axis=1)

# Reshaping and manipulation
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flattened = reshaped.flatten()
raveled = reshaped.ravel()

# Advanced indexing
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_0 = arr[0, :]
col_1 = arr[:, 1]
subarray = arr[0:2, 1:3]
bool_idx = arr > 5
filtered = arr[bool_idx]

# Broadcasting
a = np.array([[1, 2, 3]])  # Shape: (1, 3)
b = np.array([[1], [2], [3]])  # Shape: (3, 1)
result = a + b  # Shape: (3, 3)

# Useful AI/ML operations
# Normalizing data
data = np.random.randn(100, 5)
normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# One-hot encoding
labels = np.array([0, 1, 2, 1, 0])
n_classes = 3
one_hot = np.eye(n_classes)[labels]

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([[2.0, 1.0, 0.1]])
probabilities = softmax(logits)

# Batching data
def create_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]

# Example usage
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)
for X_batch, y_batch in create_batches(X, y, 32):
    # Training step would go here
    pass
```

### Example 3: Pandas for Data Manipulation
```python
import pandas as pd
import numpy as np

# DataFrame creation (10+ methods)
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
df3 = pd.read_csv('data.csv')
df4 = pd.DataFrame.from_dict({'A': [1, 2], 'B': [3, 4]})
df5 = pd.DataFrame.from_records([{'A': 1, 'B': 2}, {'A': 3, 'B': 4}])

# Series operations
s = pd.Series([1, 2, 3, 4, 5])
s_squared = s ** 2
s_filtered = s[s > 2]
s_normalized = (s - s.mean()) / s.std()

# DataFrame operations
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 70000],
    'department': ['AI', 'ML', 'AI', 'Data', 'ML']
})

# Selection and indexing
name_column = df['name']
multiple_cols = df[['name', 'age']]
row_0 = df.iloc[0]
row_by_condition = df[df['age'] > 30]
loc_selection = df.loc[df['age'] > 30, ['name', 'salary']]

# Data manipulation
df_sorted = df.sort_values('salary', ascending=False)
df_filtered = df[df['department'] == 'AI']
df['bonus'] = df['salary'] * 0.1
df['senior'] = df['age'] > 30

# Grouping and aggregation
dept_stats = df.groupby('department').agg({
    'salary': ['mean', 'min', 'max'],
    'age': 'mean'
})

dept_avg_salary = df.groupby('department')['salary'].mean()

# Data cleaning
# Handle missing values
df_with_na = df.copy()
df_with_na.loc[0, 'salary'] = np.nan

# Drop NaN
df_dropped = df_with_na.dropna()

# Fill NaN
df_filled = df_with_na.fillna(df_with_na['salary'].mean())

# Forward fill
df_ffill = df_with_na.fillna(method='ffill')

# Merging and joining
df_left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df_right = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

merged_inner = pd.merge(df_left, df_right, on='key', how='inner')
merged_outer = pd.merge(df_left, df_right, on='key', how='outer')
merged_left = pd.merge(df_left, df_right, on='key', how='left')

# Concatenation
df_concat = pd.concat([df_left, df_right], ignore_index=True)

# Apply functions
df['salary_category'] = df['salary'].apply(
    lambda x: 'High' if x > 65000 else 'Low'
)

# Map values
dept_map = {'AI': 'Artificial Intelligence', 'ML': 'Machine Learning'}
df['dept_full'] = df['department'].map(dept_map)

# Time series operations
dates = pd.date_range('2024-01-01', periods=100)
ts_df = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(100).cumsum()
})
ts_df.set_index('date', inplace=True)

# Resampling
monthly_avg = ts_df.resample('M').mean()
weekly_sum = ts_df.resample('W').sum()

# Rolling windows
rolling_mean = ts_df['value'].rolling(window=7).mean()
rolling_std = ts_df['value'].rolling(window=7).std()

# Data I/O
df.to_csv('output.csv', index=False)
df.to_json('output.json')
df.to_excel('output.xlsx', index=False)
df.to_parquet('output.parquet')

# Reading with options
df_read = pd.read_csv('data.csv',
                       sep=',',
                       header=0,
                       names=['col1', 'col2'],
                       dtype={'col1': str, 'col2': float},
                       parse_dates=['date_col'])
```

### Example 4: Data Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Example 1: Line plots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure()
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Sine and Cosine Functions')
plt.legend()
plt.grid(True)
plt.savefig('line_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Example 2: Scatter plots
n = 100
x = np.random.randn(n)
y = 2 * x + np.random.randn(n) * 0.5

plt.figure()
plt.scatter(x, y, alpha=0.5, s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Correlation')

# Add regression line
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', linewidth=2, label=f'y = {m:.2f}x + {b:.2f}')
plt.legend()
plt.show()

# Example 3: Histograms
data = np.random.randn(1000)

plt.figure()
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normal Distribution')
plt.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
plt.legend()
plt.show()

# Example 4: Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('Sine Wave')

# Plot 2: Scatter
axes[0, 1].scatter(x, y)
axes[0, 1].set_title('Scatter')

# Plot 3: Histogram
axes[1, 0].hist(data, bins=30)
axes[1, 0].set_title('Histogram')

# Plot 4: Box plot
axes[1, 1].boxplot([np.random.randn(100) for _ in range(5)])
axes[1, 1].set_title('Box Plot')

plt.tight_layout()
plt.show()

# Example 5: Seaborn visualizations
# Create sample dataset
df = pd.DataFrame({
    'feature1': np.random.randn(200),
    'feature2': np.random.randn(200),
    'category': np.random.choice(['A', 'B', 'C'], 200),
    'value': np.random.rand(200) * 100
})

# Pair plot
sns.pairplot(df, hue='category')
plt.show()

# Heatmap (correlation matrix)
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Box plot by category
plt.figure()
sns.boxplot(data=df, x='category', y='value')
plt.title('Value Distribution by Category')
plt.show()

# Violin plot
plt.figure()
sns.violinplot(data=df, x='category', y='feature1')
plt.title('Feature1 Distribution by Category')
plt.show()

# Joint plot
sns.jointplot(data=df, x='feature1', y='feature2', kind='scatter', alpha=0.5)
plt.show()

# Distribution plot
plt.figure()
sns.histplot(data=df, x='feature1', kde=True, bins=30)
plt.title('Feature1 Distribution with KDE')
plt.show()

# Count plot
plt.figure()
sns.countplot(data=df, x='category')
plt.title('Category Counts')
plt.show()
```

---

## Hands-On Exercises

### Exercise 1: NumPy Array Operations
```python
# Create a 5x5 matrix of random numbers
# Normalize each column to have mean=0 and std=1
# Find the correlation matrix
# Extract the diagonal elements
# Your code here
```

### Exercise 2: Pandas Data Cleaning
```python
# Load a messy dataset
# Handle missing values
# Remove duplicates
# Convert data types
# Create new features
# Export cleaned data
# Your code here
```

### Exercise 3: Visualization Challenge
```python
# Create a dashboard with 4 subplots:
# 1. Time series with trend line
# 2. Distribution histogram
# 3. Scatter plot with categories
# 4. Heatmap of correlations
# Your code here
```

---

## Mini Projects

### Project 1: Data Preprocessing Pipeline
Build a reusable data preprocessing class:
```python
class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def fit_transform_numeric(self, df, columns):
        # Normalize numeric columns
        pass

    def fit_transform_categorical(self, df, columns):
        # One-hot encode categorical columns
        pass

    def handle_missing_values(self, df, strategy='mean'):
        # Fill missing values
        pass

    def remove_outliers(self, df, columns, n_std=3):
        # Remove outliers using z-score
        pass
```

### Project 2: EDA (Exploratory Data Analysis) Tool
Create an automated EDA function:
```python
def perform_eda(df):
    """
    Automated exploratory data analysis:
    - Dataset summary
    - Missing value report
    - Numeric column statistics
    - Categorical column distributions
    - Correlation heatmap
    - Distribution plots
    """
    # Your implementation here
    pass
```

### Project 3: Custom Visualization Library
Build wrapper functions for common ML visualizations:
```python
def plot_learning_curves(train_scores, val_scores):
    # Plot training vs validation metrics
    pass

def plot_confusion_matrix(y_true, y_pred, labels):
    # Create confusion matrix heatmap
    pass

def plot_feature_importance(feature_names, importance_values):
    # Bar plot of feature importances
    pass
```

---

## Resources

### Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

### Interactive Tutorials
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Tutorial](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Cheat Sheets
- [NumPy Cheat Sheet](https://images.datacamp.com/image/upload/v1676302204/Marketing/Blog/Numpy_Cheat_Sheet.pdf)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)

---

## Week 1 Completion Checklist

### Technical Skills
- [ ] Can create and manipulate NumPy arrays
- [ ] Understand broadcasting rules
- [ ] Can perform matrix operations
- [ ] Master Pandas DataFrames
- [ ] Can clean and transform data
- [ ] Create professional visualizations
- [ ] Write efficient Python code
- [ ] Use Jupyter notebooks effectively

### Projects Completed
- [ ] Data preprocessing pipeline
- [ ] EDA automation tool
- [ ] Custom visualization library
- [ ] All 15+ examples executed
- [ ] All exercises completed

### Self-Assessment (1-5)
- Python fundamentals: ___/5
- NumPy proficiency: ___/5
- Pandas skills: ___/5
- Visualization: ___/5
- Code quality: ___/5

**Goal**: All areas 3+ (4+ recommended)

---

**Next**: [Week 2: Linear Algebra](../Week_2_LinearAlgebra/README.md)

---

*Time: 15-20 hours*
*Examples: 50+*
*Projects: 3*
