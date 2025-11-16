"""
Logistic Regression from Scratch
=================================
Complete implementation with binary and multi-class classification.

Topics:
1. Binary Logistic Regression
2. Sigmoid Function
3. Cross-Entropy Loss
4. Multi-class Classification (One-vs-Rest)
5. Softmax Regression
6. Evaluation Metrics
7. Decision Boundaries
8. Real-world Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("LOGISTIC REGRESSION FROM SCRATCH")
print("=" * 80)

# ==============================================================================
# EXAMPLE 1: Binary Logistic Regression
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: Binary Logistic Regression")
print("=" * 80)


class LogisticRegression:
    """Binary Logistic Regression using gradient descent."""

    def __init__(self, learning_rate=0.01, iterations=1000, regularization=None, lambda_=0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss."""
        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # Add regularization
        if self.regularization == 'l2':
            loss += (self.lambda_ / 2) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            loss += self.lambda_ * np.sum(np.abs(self.weights))

        return loss

    def fit(self, X, y):
        """Train the model."""
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)

            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Add regularization to gradients
            if self.regularization == 'l2':
                dw += self.lambda_ * self.weights
            elif self.regularization == 'l1':
                dw += self.lambda_ * np.sign(self.weights)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """Predict probabilities."""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        """Calculate accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Generate binary classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Visualize decision boundary
plt.figure(figsize=(15, 5))

# Plot 1: Decision boundary
plt.subplot(1, 3, 1)
h = 0.02
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train,
            cmap='RdYlBu', edgecolors='black', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Decision Boundary\nAccuracy: {test_acc:.4f}')

# Plot 2: Training loss
plt.subplot(1, 3, 2)
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot 3: Probability distribution
plt.subplot(1, 3, 3)
probs = model.predict_proba(X_test_scaled)
plt.hist(probs[y_test == 0], bins=20, alpha=0.5, label='Class 0', color='blue')
plt.hist(probs[y_test == 1], bins=20, alpha=0.5, label='Class 1', color='red')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Probability Distribution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('logistic_regression_binary.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved as 'logistic_regression_binary.png'")

# ==============================================================================
# EXAMPLE 2: Evaluation Metrics
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Evaluation Metrics")
print("=" * 80)


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp], [fn, tp]])


def classification_metrics(y_true, y_pred):
    """Compute precision, recall, F1-score."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


# Get predictions
y_pred = model.predict(X_test_scaled)

# Compute metrics
metrics = classification_metrics(y_test, y_pred)

print("Classification Metrics:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1-Score:  {metrics['f1_score']:.4f}")

print("\nConfusion Matrix:")
print(metrics['confusion_matrix'])
print("           Predicted")
print("           0    1")
print(f"Actual 0  {metrics['confusion_matrix'][0, 0]:3d}  {metrics['confusion_matrix'][0, 1]:3d}")
print(f"       1  {metrics['confusion_matrix'][1, 0]:3d}  {metrics['confusion_matrix'][1, 1]:3d}")

# ==============================================================================
# EXAMPLE 3: Multi-class Classification (Softmax Regression)
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Multi-class Classification (Softmax)")
print("=" * 80)


class SoftmaxRegression:
    """Multi-class logistic regression using softmax."""

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def softmax(self, z):
        """Softmax function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        """Compute cross-entropy loss."""
        n_samples = y_true.shape[0]
        # Clip predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / n_samples
        return loss

    def one_hot_encode(self, y, n_classes):
        """One-hot encode labels."""
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def fit(self, X, y):
        """Train the model."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize parameters
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        # One-hot encode labels
        y_encoded = self.one_hot_encode(y, n_classes)

        for i in range(self.iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(linear_pred)

            # Compute loss
            loss = self.cross_entropy_loss(y_encoded, y_pred)
            self.losses.append(loss)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y_encoded))
            db = (1/n_samples) * np.sum(y_pred - y_encoded, axis=0)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 200 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """Predict class probabilities."""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.softmax(linear_pred)

    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """Calculate accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
softmax_model = SoftmaxRegression(learning_rate=0.1, iterations=1000)
softmax_model.fit(X_train_scaled, y_train)

# Evaluate
train_acc = softmax_model.score(X_train_scaled, y_train)
test_acc = softmax_model.score(X_test_scaled, y_test)

print(f"\nMulti-class Classification Results:")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Per-class accuracy
y_pred = softmax_model.predict(X_test_scaled)
print("\nPer-class Accuracy:")
for i, class_name in enumerate(iris.target_names):
    mask = y_test == i
    if np.sum(mask) > 0:
        class_acc = np.mean(y_pred[mask] == y_test[mask])
        print(f"  {class_name}: {class_acc:.4f}")

# ==============================================================================
# EXAMPLE 4: Breast Cancer Classification (Real Dataset)
# ==============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: Breast Cancer Classification")
print("=" * 80)

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {cancer.target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with regularization
model_l2 = LogisticRegression(learning_rate=0.1, iterations=2000,
                               regularization='l2', lambda_=0.01)
model_l2.fit(X_train_scaled, y_train)

# Evaluate
train_acc = model_l2.score(X_train_scaled, y_train)
test_acc = model_l2.score(X_test_scaled, y_test)

y_pred = model_l2.predict(X_test_scaled)
metrics = classification_metrics(y_test, y_pred)

print(f"\nResults with L2 Regularization:")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Feature importance
feature_importance = np.abs(model_l2.weights)
top_features_idx = np.argsort(feature_importance)[::-1][:10]

print("\nTop 10 Most Important Features:")
for i, idx in enumerate(top_features_idx):
    print(f"  {i+1}. {cancer.feature_names[idx]:30s}: {feature_importance[idx]:.4f}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY: Logistic Regression Complete!")
print("=" * 80)
print("""
Key Concepts Covered:
1. ✓ Binary logistic regression from scratch
2. ✓ Sigmoid activation function
3. ✓ Cross-entropy loss
4. ✓ Multi-class classification with softmax
5. ✓ Evaluation metrics (accuracy, precision, recall, F1)
6. ✓ Confusion matrix analysis
7. ✓ L1 and L2 regularization
8. ✓ Real-world applications (Iris, Breast Cancer)

Key Takeaways:
- Logistic regression: Classification algorithm
- Sigmoid: Maps to [0, 1] for binary classification
- Softmax: Extends to multi-class problems
- Cross-entropy: Proper loss for classification
- Regularization: Prevents overfitting
- Metrics matter: Choose based on problem (e.g., F1 for imbalanced data)

Next: Decision Trees and Random Forests!
""")
