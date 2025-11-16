"""
Linear Regression from Scratch
================================
Complete implementation of linear regression with multiple examples.

Topics:
1. Simple Linear Regression
2. Multiple Linear Regression
3. Gradient Descent
4. Normal Equation
5. Polynomial Regression
6. Ridge Regression (L2)
7. Lasso Regression (L1)
8. Real-world Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("LINEAR REGRESSION FROM SCRATCH")
print("=" * 80)

# ==============================================================================
# EXAMPLE 1: Simple Linear Regression (One Feature)
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: Simple Linear Regression")
print("=" * 80)


class SimpleLinearRegression:
    """Simple Linear Regression: y = mx + b"""

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.m = 0  # slope
        self.b = 0  # intercept
        self.losses = []

    def fit(self, X, y):
        """Train the model using gradient descent."""
        n = len(X)

        for i in range(self.iterations):
            # Predictions
            y_pred = self.m * X + self.b

            # Calculate loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)

            # Calculate gradients
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            # Update parameters
            self.m -= self.lr * dm
            self.b -= self.lr * db

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}, m = {self.m:.4f}, b = {self.b:.4f}")

    def predict(self, X):
        """Make predictions."""
        return self.m * X + self.b

    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100)
y = 4 + 3 * X + np.random.randn(100)

# Train model
model = SimpleLinearRegression(learning_rate=0.1, iterations=1000)
model.fit(X, y)

# Evaluate
r2 = model.score(X, y)
print(f"\nFinal R² Score: {r2:.4f}")
print(f"Final equation: y = {model.m:.2f}x + {model.b:.2f}")
print(f"True equation: y = 3.00x + 4.00")

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('simple_linear_regression.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'simple_linear_regression.png'")

# ==============================================================================
# EXAMPLE 2: Multiple Linear Regression (Vector Form)
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Multiple Linear Regression")
print("=" * 80)


class MultipleLinearRegression:
    """Multiple Linear Regression: y = X @ w + b"""

    def __init__(self, learning_rate=0.01, iterations=1000, regularization=None, lambda_=0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.regularization = regularization  # 'l1', 'l2', or None
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        """Train using gradient descent with optional regularization."""
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute loss
            mse_loss = np.mean((y - y_pred) ** 2)

            # Add regularization to loss
            if self.regularization == 'l2':
                reg_loss = self.lambda_ * np.sum(self.weights ** 2)
                loss = mse_loss + reg_loss
            elif self.regularization == 'l1':
                reg_loss = self.lambda_ * np.sum(np.abs(self.weights))
                loss = mse_loss + reg_loss
            else:
                loss = mse_loss

            self.losses.append(loss)

            # Compute gradients
            dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2/n_samples) * np.sum(y - y_pred)

            # Add regularization to gradients
            if self.regularization == 'l2':
                dw += 2 * self.lambda_ * self.weights
            elif self.regularization == 'l1':
                dw += self.lambda_ * np.sign(self.weights)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 200 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


# Generate multi-feature dataset
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = MultipleLinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
train_r2 = model.score(X_train_scaled, y_train)
test_r2 = model.score(X_test_scaled, y_test)

print(f"\nTraining R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")

# ==============================================================================
# EXAMPLE 3: Normal Equation (Closed-form Solution)
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Normal Equation Solution")
print("=" * 80)


class LinearRegressionNormalEquation:
    """Linear Regression using Normal Equation: w = (X^T X)^(-1) X^T y"""

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Solve using normal equation."""
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal equation: theta = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        """Make predictions."""
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


# Train using normal equation
model_ne = LinearRegressionNormalEquation()
model_ne.fit(X_train_scaled, y_train)

# Evaluate
train_r2_ne = model_ne.score(X_train_scaled, y_train)
test_r2_ne = model_ne.score(X_test_scaled, y_test)

print(f"Normal Equation - Training R²: {train_r2_ne:.4f}")
print(f"Normal Equation - Test R²: {test_r2_ne:.4f}")

print("\nComparison:")
print(f"Gradient Descent R²: {test_r2:.4f}")
print(f"Normal Equation R²: {test_r2_ne:.4f}")
print(f"Difference: {abs(test_r2 - test_r2_ne):.6f}")

# ==============================================================================
# EXAMPLE 4: Polynomial Regression
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: Polynomial Regression")
print("=" * 80)


def create_polynomial_features(X, degree):
    """Create polynomial features up to given degree."""
    n_samples = X.shape[0]
    features = [X]

    for d in range(2, degree + 1):
        features.append(X ** d)

    return np.column_stack(features)


# Generate non-linear data
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.5

# Try different polynomial degrees
degrees = [1, 2, 3, 5]
plt.figure(figsize=(15, 4))

for idx, degree in enumerate(degrees):
    # Create polynomial features
    X_poly = create_polynomial_features(X, degree)

    # Train model
    model_poly = LinearRegressionNormalEquation()
    model_poly.fit(X_poly, y.ravel())

    # Predictions
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_test_poly = create_polynomial_features(X_test, degree)
    y_pred = model_poly.predict(X_test_poly)

    # Calculate R²
    r2 = model_poly.score(X_poly, y.ravel())

    # Plot
    plt.subplot(1, 4, idx + 1)
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Degree {degree}\nR² = {r2:.4f}')
    plt.grid(True)

plt.tight_layout()
plt.savefig('polynomial_regression.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'polynomial_regression.png'")

# ==============================================================================
# EXAMPLE 5: Ridge Regression (L2 Regularization)
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 5: Ridge Regression (L2 Regularization)")
print("=" * 80)

# Generate data with more features than needed
X, y = make_regression(n_samples=100, n_features=50, n_informative=10,
                       noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different regularization strengths
lambdas = [0, 0.01, 0.1, 1.0, 10.0]
results = []

for lambda_ in lambdas:
    model = MultipleLinearRegression(
        learning_rate=0.01,
        iterations=1000,
        regularization='l2',
        lambda_=lambda_
    )
    model.fit(X_train_scaled, y_train)

    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)

    results.append({
        'lambda': lambda_,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'weights_norm': np.linalg.norm(model.weights)
    })

    print(f"λ = {lambda_:6.2f}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}, ||w|| = {np.linalg.norm(model.weights):.2f}")

# ==============================================================================
# EXAMPLE 6: Real-World Application - Diabetes Dataset
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 6: Real-World Application - Diabetes Progression")
print("=" * 80)

# Load diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {diabetes.feature_names}")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = MultipleLinearRegression(learning_rate=0.01, iterations=2000)
model.fit(X_train_scaled, y_train)

# Evaluate
train_r2 = model.score(X_train_scaled, y_train)
test_r2 = model.score(X_test_scaled, y_test)

# Calculate other metrics
y_pred = model.predict(X_test_scaled)
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))

print(f"\nResults:")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Feature importance (absolute value of weights)
feature_importance = np.abs(model.weights)
sorted_idx = np.argsort(feature_importance)[::-1]

print("\nTop 5 Most Important Features:")
for i in range(5):
    idx = sorted_idx[i]
    print(f"  {i+1}. {diabetes.feature_names[idx]:10s}: {feature_importance[idx]:.4f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Predictions vs Actual\nR² = {test_r2:.4f}')
plt.grid(True)

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)

plt.tight_layout()
plt.savefig('diabetes_regression.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'diabetes_regression.png'")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY: Linear Regression Complete!")
print("=" * 80)
print("""
Key Concepts Covered:
1. ✓ Simple linear regression from scratch
2. ✓ Multiple linear regression with gradient descent
3. ✓ Normal equation (closed-form solution)
4. ✓ Polynomial regression for non-linear data
5. ✓ Ridge regression (L2 regularization)
6. ✓ Real-world application on diabetes dataset
7. ✓ Model evaluation (R², MSE, RMSE, MAE)
8. ✓ Feature importance analysis

Key Takeaways:
- Gradient descent: Iterative optimization
- Normal equation: Direct solution (small datasets)
- Polynomial features: Capture non-linearity
- Regularization: Prevent overfitting
- Feature scaling: Essential for gradient descent
- R² score: Measures goodness of fit

Next: Logistic Regression for Classification!
""")
