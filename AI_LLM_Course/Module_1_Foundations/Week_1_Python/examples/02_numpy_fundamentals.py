"""
NumPy Fundamentals for AI/ML
=============================
100+ Examples of NumPy operations essential for AI development.

Topics:
1. Array Creation (15+ methods)
2. Array Indexing and Slicing
3. Array Operations
4. Broadcasting
5. Linear Algebra
6. Statistical Operations
7. Random Number Generation
8. Advanced Techniques for ML
"""

import numpy as np
import time

print("=" * 80)
print("NumPy for AI/ML - Comprehensive Guide")
print("=" * 80)

# ==============================================================================
# SECTION 1: ARRAY CREATION (15+ Methods)
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 1: Array Creation Methods")
print("=" * 80)

# Method 1: From Python lists
arr1 = np.array([1, 2, 3, 4, 5])
print(f"1. From list: {arr1}")

# Method 2: From nested lists (2D array)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2. 2D array:\n{arr2}")

# Method 3: Zeros
arr3 = np.zeros((3, 4))
print(f"3. Zeros (3x4):\n{arr3}")

# Method 4: Ones
arr4 = np.ones((2, 3))
print(f"4. Ones (2x3):\n{arr4}")

# Method 5: Full (filled with specific value)
arr5 = np.full((2, 2), 7)
print(f"5. Full (2x2, value=7):\n{arr5}")

# Method 6: Identity matrix
arr6 = np.eye(4)
print(f"6. Identity (4x4):\n{arr6}")

# Method 7: Arange (like Python range)
arr7 = np.arange(0, 10, 2)
print(f"7. Arange (0 to 10, step 2): {arr7}")

# Method 8: Linspace (evenly spaced)
arr8 = np.linspace(0, 1, 5)
print(f"8. Linspace (0 to 1, 5 points): {arr8}")

# Method 9: Logspace (logarithmically spaced)
arr9 = np.logspace(0, 2, 5)
print(f"9. Logspace (10^0 to 10^2, 5 points): {arr9}")

# Method 10: Random uniform
arr10 = np.random.rand(3, 3)
print(f"10. Random uniform (3x3):\n{arr10}")

# Method 11: Random normal (Gaussian)
arr11 = np.random.randn(3, 3)
print(f"11. Random normal (3x3):\n{arr11}")

# Method 12: Random integers
arr12 = np.random.randint(0, 100, (3, 3))
print(f"12. Random integers 0-100 (3x3):\n{arr12}")

# Method 13: Empty (uninitialized)
arr13 = np.empty((2, 2))
print(f"13. Empty (2x2):\n{arr13}")

# Method 14: Like (same shape as another array)
arr14 = np.zeros_like(arr2)
print(f"14. Zeros like arr2:\n{arr14}")

# Method 15: From function
arr15 = np.fromfunction(lambda i, j: i + j, (3, 3))
print(f"15. From function (i+j):\n{arr15}")

# ==============================================================================
# SECTION 2: Array Properties and Information
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 2: Array Properties")
print("=" * 80)

arr = np.random.randn(3, 4, 5)

print(f"Shape: {arr.shape}")  # Dimensions
print(f"Size: {arr.size}")  # Total elements
print(f"Ndim: {arr.ndim}")  # Number of dimensions
print(f"Dtype: {arr.dtype}")  # Data type
print(f"Itemsize: {arr.itemsize}")  # Size of each element in bytes
print(f"Nbytes: {arr.nbytes}")  # Total bytes

# ==============================================================================
# SECTION 3: Array Indexing and Slicing
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 3: Indexing and Slicing")
print("=" * 80)

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(f"Original array:\n{arr}\n")

# Basic indexing
print(f"Element [0, 0]: {arr[0, 0]}")
print(f"Element [1, 2]: {arr[1, 2]}")

# Row slicing
print(f"First row: {arr[0, :]}")
print(f"Last row: {arr[-1, :]}")

# Column slicing
print(f"First column: {arr[:, 0]}")
print(f"Last column: {arr[:, -1]}")

# Subarray
print(f"Subarray [0:2, 1:3]:\n{arr[0:2, 1:3]}")

# Boolean indexing (very important for ML!)
mask = arr > 5
print(f"\nBoolean mask (>5):\n{mask}")
print(f"Elements >5: {arr[mask]}")

# Fancy indexing
rows = np.array([0, 2])
cols = np.array([1, 3])
print(f"Fancy indexing [(0,1), (2,3)]: {arr[rows, cols]}")

# ==============================================================================
# SECTION 4: Array Operations (Element-wise)
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 4: Element-wise Operations")
print("=" * 80)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(f"Array a:\n{a}\n")
print(f"Array b:\n{b}\n")

# Arithmetic operations
print(f"Addition a + b:\n{a + b}\n")
print(f"Subtraction a - b:\n{a - b}\n")
print(f"Multiplication a * b:\n{a * b}\n")
print(f"Division a / b:\n{a / b}\n")
print(f"Power a ** 2:\n{a ** 2}\n")
print(f"Square root √a:\n{np.sqrt(a)}\n")

# Scalar operations
print(f"Add scalar a + 10:\n{a + 10}\n")
print(f"Multiply scalar a * 2:\n{a * 2}\n")

# Mathematical functions
print(f"Exponential e^a:\n{np.exp(a)}\n")
print(f"Logarithm log(a):\n{np.log(a)}\n")
print(f"Sin(a):\n{np.sin(a)}\n")
print(f"Absolute |a|:\n{np.abs(a)}\n")

# ==============================================================================
# SECTION 5: Broadcasting (Critical for ML!)
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 5: Broadcasting")
print("=" * 80)

# Example 1: Vector + Scalar
v = np.array([1, 2, 3])
print(f"Vector: {v}")
print(f"Vector + 10: {v + 10}")

# Example 2: Matrix + Vector (row-wise)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row_vector = np.array([10, 20, 30])

print(f"\nMatrix:\n{matrix}")
print(f"Row vector: {row_vector}")
print(f"Matrix + Row vector:\n{matrix + row_vector}")

# Example 3: Matrix + Column Vector
col_vector = np.array([[10], [20], [30]])
print(f"\nColumn vector:\n{col_vector}")
print(f"Matrix + Column vector:\n{matrix + col_vector}")

# Example 4: Broadcasting in action (normalize columns)
data = np.random.randn(5, 3)
mean = data.mean(axis=0)  # Mean of each column
std = data.std(axis=0)  # Std of each column
normalized = (data - mean) / std  # Broadcasting!

print(f"\nData:\n{data}")
print(f"Mean: {mean}")
print(f"Std: {std}")
print(f"Normalized:\n{normalized}")

# ==============================================================================
# SECTION 6: Linear Algebra Operations
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 6: Linear Algebra")
print("=" * 80)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

print(f"Matrix A:\n{A}\n")
print(f"Matrix B:\n{B}\n")
print(f"Vector v: {v}\n")

# Matrix multiplication
print(f"Matrix multiplication A @ B:\n{np.dot(A, B)}\n")
print(f"Alternative A @ B:\n{A @ B}\n")

# Matrix-vector multiplication
print(f"Matrix-vector A @ v: {np.dot(A, v)}\n")

# Transpose
print(f"Transpose A^T:\n{A.T}\n")

# Determinant
det = np.linalg.det(A)
print(f"Determinant |A|: {det}\n")

# Inverse
inv = np.linalg.inv(A)
print(f"Inverse A^(-1):\n{inv}\n")

# Verify: A @ A^(-1) = I
identity = np.dot(A, inv)
print(f"Verification A @ A^(-1):\n{identity}\n")

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}\n")

# Singular Value Decomposition (SVD)
U, s, Vt = np.linalg.svd(A)
print(f"SVD of A:")
print(f"U:\n{U}")
print(f"Singular values: {s}")
print(f"V^T:\n{Vt}\n")

# Matrix rank
rank = np.linalg.matrix_rank(A)
print(f"Rank of A: {rank}\n")

# ==============================================================================
# SECTION 7: Statistical Operations
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 7: Statistical Operations")
print("=" * 80)

data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

print(f"Data:\n{data}\n")

# Basic statistics
print(f"Mean (all): {np.mean(data)}")
print(f"Mean (axis=0): {np.mean(data, axis=0)}")  # Column means
print(f"Mean (axis=1): {np.mean(data, axis=1)}")  # Row means

print(f"\nMedian: {np.median(data)}")
print(f"Std: {np.std(data)}")
print(f"Variance: {np.var(data)}")

print(f"\nMin: {np.min(data)}")
print(f"Max: {np.max(data)}")
print(f"Sum: {np.sum(data)}")
print(f"Product: {np.prod(data)}")

# Percentiles
print(f"\n25th percentile: {np.percentile(data, 25)}")
print(f"50th percentile: {np.percentile(data, 50)}")
print(f"75th percentile: {np.percentile(data, 75)}")

# Correlation
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
correlation = np.corrcoef(x, y)
print(f"\nCorrelation matrix:\n{correlation}")

# ==============================================================================
# SECTION 8: Array Manipulation
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 8: Array Manipulation")
print("=" * 80)

arr = np.arange(12)
print(f"Original: {arr}")

# Reshape
reshaped = arr.reshape(3, 4)
print(f"Reshaped (3x4):\n{reshaped}")

# Flatten
flattened = reshaped.flatten()
print(f"Flattened: {flattened}")

# Transpose
transposed = reshaped.T
print(f"Transposed:\n{transposed}")

# Concatenate
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

concat_axis0 = np.concatenate([a, b], axis=0)
concat_axis1 = np.concatenate([a, b], axis=1)

print(f"\nConcatenate axis=0:\n{concat_axis0}")
print(f"Concatenate axis=1:\n{concat_axis1}")

# Stack
vstacked = np.vstack([a, b])
hstacked = np.hstack([a, b])

print(f"\nVStack:\n{vstacked}")
print(f"HStack:\n{hstacked}")

# Split
arr = np.arange(12).reshape(4, 3)
split_arrays = np.split(arr, 2, axis=0)
print(f"\nSplit into 2:\n{split_arrays[0]}\n{split_arrays[1]}")

# ==============================================================================
# SECTION 9: Advanced ML Operations
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 9: ML-Specific Operations")
print("=" * 80)

# 1. Softmax function (for neural network outputs)
def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
probabilities = softmax(logits)
print(f"Logits:\n{logits}")
print(f"Probabilities (softmax):\n{probabilities}")
print(f"Sum of probs: {probabilities.sum(axis=1)}")  # Should be 1

# 2. One-hot encoding (for labels)
labels = np.array([0, 1, 2, 1, 0])
n_classes = 3
one_hot = np.eye(n_classes)[labels]
print(f"\nLabels: {labels}")
print(f"One-hot encoded:\n{one_hot}")

# 3. Train/test split
def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Test split function
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f"\nDataset split:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")

# 4. Batch creation
def create_batches(X, y, batch_size=32):
    """Create batches for training."""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        batches.append((X[batch_idx], y[batch_idx]))

    return batches

batches = create_batches(X_train, y_train, batch_size=32)
print(f"\nNumber of batches: {len(batches)}")
print(f"Batch 0 shape: X={batches[0][0].shape}, y={batches[0][1].shape}")

# 5. Feature normalization
def normalize_features(X, method='standard'):
    """Normalize features."""
    if method == 'standard':
        # Zero mean, unit variance
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / (std + 1e-8)
    elif method == 'minmax':
        # Scale to [0, 1]
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        return (X - min_val) / (max_val - min_val + 1e-8)

X_normalized = normalize_features(X_train, method='standard')
print(f"\nNormalized features:")
print(f"  Mean: {X_normalized.mean(axis=0)}")
print(f"  Std: {X_normalized.std(axis=0)}")

# 6. Cosine similarity (for embeddings)
def cosine_similarity(a, b):
    """Compute cosine similarity between vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])
vec3 = np.array([-1, -2, -3])

print(f"\nCosine similarity:")
print(f"  vec1 vs vec2: {cosine_similarity(vec1, vec2):.4f}")
print(f"  vec1 vs vec3: {cosine_similarity(vec1, vec3):.4f}")

# ==============================================================================
# SECTION 10: Performance Tips
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 10: Performance Comparison")
print("=" * 80)

# NumPy vs Python lists
n = 1000000

# Python list
python_list = list(range(n))
start = time.time()
python_squared = [x ** 2 for x in python_list]
python_time = time.time() - start

# NumPy array
numpy_array = np.arange(n)
start = time.time()
numpy_squared = numpy_array ** 2
numpy_time = time.time() - start

print(f"Squaring {n:,} numbers:")
print(f"  Python list: {python_time:.4f} seconds")
print(f"  NumPy array: {numpy_time:.4f} seconds")
print(f"  Speedup: {python_time / numpy_time:.1f}x")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("Summary: NumPy Fundamentals Complete!")
print("=" * 80)
print("""
Key Concepts Covered:
1. ✓ 15+ array creation methods
2. ✓ Indexing, slicing, and masking
3. ✓ Element-wise operations
4. ✓ Broadcasting (critical for ML!)
5. ✓ Linear algebra operations
6. ✓ Statistical functions
7. ✓ Array manipulation
8. ✓ ML-specific operations (softmax, one-hot, etc.)
9. ✓ Performance optimization

NumPy is 10-100x faster than Python lists for numerical operations!

Next: Pandas for data manipulation!
""")
