"""
Python Essentials for AI/ML
============================
Comprehensive examples of Python fundamentals needed for AI development.

Topics:
1. Data Structures
2. List Comprehensions
3. Functions and Lambda
4. Object-Oriented Programming
5. File I/O
6. Error Handling
7. Iterators and Generators
"""

# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================

print("=" * 80)
print("EXAMPLE 1: Lists - The Most Versatile Data Structure")
print("=" * 80)

# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, [1, 2, 3]]
empty = []

# List operations
numbers.append(6)  # Add to end
numbers.insert(0, 0)  # Insert at position
numbers.extend([7, 8, 9])  # Add multiple
popped = numbers.pop()  # Remove and return last
numbers.remove(0)  # Remove first occurrence

# List slicing (crucial for data manipulation)
first_three = numbers[:3]
last_three = numbers[-3:]
every_second = numbers[::2]
reversed_list = numbers[::-1]

print(f"Original: {numbers}")
print(f"First three: {first_three}")
print(f"Last three: {last_three}")
print(f"Every second: {every_second}")
print(f"Reversed: {reversed_list}")

# ==============================================================================
# 2. DICTIONARIES - For Configuration and Data Storage
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Dictionaries for Model Configuration")
print("=" * 80)

# Model hyperparameters (common use case in ML)
model_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam',
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'dropout': 0.2
}

# Accessing values
lr = model_config['learning_rate']
optimizer = model_config.get('optimizer', 'sgd')  # With default

# Modifying
model_config['learning_rate'] = 0.0001
model_config.update({'momentum': 0.9, 'weight_decay': 1e-5})

# Iterating
print("\nModel Configuration:")
for key, value in model_config.items():
    print(f"  {key}: {value}")

# Dictionary comprehension
squared_dict = {x: x**2 for x in range(1, 6)}
print(f"\nSquared numbers: {squared_dict}")

# ==============================================================================
# 3. SETS - For Unique Elements and Fast Lookups
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Sets for Data Deduplication")
print("=" * 80)

# Remove duplicates from data
data_with_duplicates = [1, 2, 2, 3, 3, 3, 4, 4, 5]
unique_data = list(set(data_with_duplicates))
print(f"Original: {data_with_duplicates}")
print(f"Unique: {unique_data}")

# Set operations (useful for feature engineering)
train_features = {'age', 'income', 'education', 'zipcode'}
test_features = {'age', 'income', 'city', 'state'}

common_features = train_features & test_features  # Intersection
all_features = train_features | test_features  # Union
only_train = train_features - test_features  # Difference

print(f"\nTrain features: {train_features}")
print(f"Test features: {test_features}")
print(f"Common features: {common_features}")
print(f"All features: {all_features}")
print(f"Only in train: {only_train}")

# ==============================================================================
# 4. LIST COMPREHENSIONS - Efficient Data Transformation
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: List Comprehensions for Data Processing")
print("=" * 80)

# Example 1: Square numbers
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squared: {squared}")

# Example 2: Filter and transform
even_squared = [x**2 for x in numbers if x % 2 == 0]
print(f"Even numbers squared: {even_squared}")

# Example 3: Nested list comprehension (for matrix operations)
matrix = [[i+j for j in range(3)] for i in range(3)]
print(f"\n3x3 Matrix:")
for row in matrix:
    print(f"  {row}")

# Example 4: Flatten nested list (common in data preprocessing)
nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(f"\nNested: {nested}")
print(f"Flattened: {flattened}")

# Example 5: Multiple conditions
data = list(range(1, 21))
filtered = [x for x in data if x % 2 == 0 and x % 3 == 0]
print(f"\nNumbers divisible by 2 AND 3: {filtered}")

# ==============================================================================
# 5. FUNCTIONS - Reusable Code Blocks
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 5: Functions for ML Operations")
print("=" * 80)

def normalize(data, min_val=0, max_val=1):
    """
    Normalize data to a specified range.

    Args:
        data (list): Input data
        min_val (float): Minimum value of range
        max_val (float): Maximum value of range

    Returns:
        list: Normalized data
    """
    data_min = min(data)
    data_max = max(data)
    range_val = data_max - data_min

    if range_val == 0:
        return [min_val] * len(data)

    normalized = [
        (x - data_min) / range_val * (max_val - min_val) + min_val
        for x in data
    ]
    return normalized


# Test normalization
raw_data = [10, 20, 30, 40, 50]
normalized_data = normalize(raw_data, 0, 1)
print(f"Raw data: {raw_data}")
print(f"Normalized (0-1): {normalized_data}")

# Lambda functions (for quick transformations)
square = lambda x: x ** 2
add = lambda x, y: x + y

print(f"\nLambda square(5): {square(5)}")
print(f"Lambda add(3, 4): {add(3, 4)}")

# Using lambda with map, filter, reduce
data = [1, 2, 3, 4, 5]
squared_map = list(map(lambda x: x**2, data))
evens = list(filter(lambda x: x % 2 == 0, data))

from functools import reduce
sum_all = reduce(lambda x, y: x + y, data)

print(f"\nMap square: {squared_map}")
print(f"Filter evens: {evens}")
print(f"Reduce sum: {sum_all}")

# ==============================================================================
# 6. OBJECT-ORIENTED PROGRAMMING - Building ML Models
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 6: OOP for Machine Learning Models")
print("=" * 80)


class LinearRegression:
    """
    Simple Linear Regression Model.
    Demonstrates OOP principles for ML.
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        """Initialize model parameters."""
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        """
        Train the model.

        Args:
            X (list): Training features
            y (list): Training labels
        """
        n_samples = len(X)
        n_features = len(X[0]) if isinstance(X[0], list) else 1

        # Initialize parameters
        self.weights = [0.0] * n_features if n_features > 1 else 0.0
        self.bias = 0.0

        print(f"Training Linear Regression:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Iterations: {self.iterations}")

    def predict(self, X):
        """Make predictions."""
        if isinstance(X[0], list):
            predictions = [
                sum(w * x for w, x in zip(self.weights, sample)) + self.bias
                for sample in X
            ]
        else:
            predictions = [self.weights * x + self.bias for x in X]
        return predictions

    def get_params(self):
        """Get model parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate
        }


# Create and use the model
model = LinearRegression(learning_rate=0.01, iterations=1000)
X_train = [[1], [2], [3], [4], [5]]
y_train = [2, 4, 6, 8, 10]

model.fit(X_train, y_train)
params = model.get_params()
print(f"\nModel parameters: {params}")

# ==============================================================================
# 7. INHERITANCE - Building Model Hierarchies
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 7: Inheritance for Model Families")
print("=" * 80)


class BaseModel:
    """Base class for all ML models."""

    def __init__(self, name):
        self.name = name
        self.is_trained = False

    def fit(self, X, y):
        raise NotImplementedError("Subclass must implement fit()")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        raise NotImplementedError("Subclass must implement predict()")

    def score(self, X, y):
        """Default scoring method."""
        predictions = self.predict(X)
        # Simple accuracy for classification
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)


class LogisticRegression(BaseModel):
    """Logistic Regression inheriting from BaseModel."""

    def __init__(self):
        super().__init__("Logistic Regression")
        self.weights = None

    def fit(self, X, y):
        print(f"\nTraining {self.name}...")
        # Training logic here
        self.is_trained = True
        print(f"✓ {self.name} training complete!")

    def predict(self, X):
        super().predict(X)  # Check if trained
        print(f"Making predictions with {self.name}...")
        # Prediction logic here
        return [0] * len(X)  # Placeholder


# Use the inherited class
lr_model = LogisticRegression()
print(f"Model name: {lr_model.name}")
print(f"Is trained: {lr_model.is_trained}")

# ==============================================================================
# 8. GENERATORS - Memory-Efficient Data Loading
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 8: Generators for Large Dataset Handling")
print("=" * 80)


def data_batch_generator(data, batch_size):
    """
    Generate batches of data (memory efficient).

    Args:
        data (list): Complete dataset
        batch_size (int): Size of each batch

    Yields:
        list: Batch of data
    """
    n_batches = len(data) // batch_size

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        yield data[start_idx:end_idx]

    # Handle remaining data
    if len(data) % batch_size != 0:
        yield data[n_batches * batch_size:]


# Example usage
dataset = list(range(100))
batch_size = 32

print(f"Dataset size: {len(dataset)}")
print(f"Batch size: {batch_size}")
print("\nBatches:")

for i, batch in enumerate(data_batch_generator(dataset, batch_size), 1):
    print(f"  Batch {i}: size={len(batch)}, range=[{batch[0]}...{batch[-1]}]")

# Generator expression (like list comprehension but lazy)
squares_gen = (x**2 for x in range(10))
print(f"\nGenerator: {squares_gen}")
print(f"First 5 squares: {[next(squares_gen) for _ in range(5)]}")

# ==============================================================================
# 9. ERROR HANDLING - Robust Code
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 9: Exception Handling in ML Pipelines")
print("=" * 80)


def safe_divide(a, b):
    """Safely divide two numbers with error handling."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print(f"Error: Cannot divide {a} by zero!")
        return None
    except TypeError as e:
        print(f"Error: Invalid types - {e}")
        return None
    finally:
        print(f"  Division operation attempted: {a} / {b}")


print("Test 1: Normal division")
print(f"Result: {safe_divide(10, 2)}")

print("\nTest 2: Division by zero")
print(f"Result: {safe_divide(10, 0)}")

print("\nTest 3: Invalid types")
print(f"Result: {safe_divide('10', 2)}")


# Custom exceptions for ML
class ModelNotTrainedError(Exception):
    """Raised when trying to use an untrained model."""
    pass


class InvalidDataShapeError(Exception):
    """Raised when data shape doesn't match expected."""
    pass


def predict_with_model(model, data):
    """Make predictions with proper error handling."""
    try:
        if not model.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before making predictions"
            )

        if len(data) == 0:
            raise InvalidDataShapeError("Data cannot be empty")

        return model.predict(data)

    except ModelNotTrainedError as e:
        print(f"Model Error: {e}")
        return None
    except InvalidDataShapeError as e:
        print(f"Data Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None


print("\n" + "=" * 80)
print("EXAMPLE 10: File I/O for Data Loading")
print("=" * 80)

# Writing data to file
data_to_save = {
    'model': 'neural_network',
    'accuracy': 0.95,
    'loss': 0.05,
    'epochs': 100
}

# Write to file
try:
    with open('model_results.txt', 'w') as f:
        for key, value in data_to_save.items():
            f.write(f"{key}: {value}\n")
    print("✓ Data written to file successfully")
except IOError as e:
    print(f"Error writing to file: {e}")

# Read from file
try:
    with open('model_results.txt', 'r') as f:
        content = f.read()
        print("\nFile content:")
        print(content)
except FileNotFoundError:
    print("File not found")
except IOError as e:
    print(f"Error reading file: {e}")

# ==============================================================================
# 11. DECORATORS - For Timing and Logging
# ==============================================================================

print("=" * 80)
print("EXAMPLE 11: Decorators for ML Operations")
print("=" * 80)

import time


def timing_decorator(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"  {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@timing_decorator
def train_model(n_iterations):
    """Simulate model training."""
    total = 0
    for i in range(n_iterations):
        total += i ** 2
    return total


print("Training model with 1,000,000 iterations:")
result = train_model(1000000)

# ==============================================================================
# 12. CONTEXT MANAGERS - For Resource Management
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 12: Context Managers for Safe Resource Handling")
print("=" * 80)


class ModelCheckpoint:
    """Context manager for saving model checkpoints."""

    def __init__(self, model_name):
        self.model_name = model_name

    def __enter__(self):
        print(f"Starting checkpoint for {self.model_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Saving checkpoint for {self.model_name}")
        if exc_type is not None:
            print(f"Error occurred: {exc_val}")
        return False  # Don't suppress exceptions


# Usage
with ModelCheckpoint("my_neural_network") as checkpoint:
    print("  Training in progress...")
    print("  Epoch 1 complete")
    print("  Epoch 2 complete")

print("\n" + "=" * 80)
print("Summary: Python Essentials Complete!")
print("=" * 80)
print("""
Key Takeaways:
1. ✓ Data structures: lists, dicts, sets, tuples
2. ✓ List comprehensions for efficient data transformation
3. ✓ Functions and lambda expressions
4. ✓ OOP for building model classes
5. ✓ Generators for memory-efficient data loading
6. ✓ Error handling for robust code
7. ✓ File I/O for data persistence
8. ✓ Decorators for timing and logging
9. ✓ Context managers for resource management

Next: NumPy for numerical computing!
""")
