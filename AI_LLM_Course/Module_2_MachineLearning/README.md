# Module 2: Machine Learning Foundations

## Overview

Master classical machine learning algorithms before diving into deep learning. Build ML models from scratch, understand the mathematics, and apply them to real-world problems.

**Duration**: 4-5 weeks (70-90 hours)
**Difficulty**: Intermediate

---

## Learning Objectives

✅ Implement ML algorithms from scratch (no sklearn)
✅ Understand the mathematics behind each algorithm
✅ Build complete ML pipelines
✅ Master feature engineering techniques
✅ Achieve 85%+ accuracy on real datasets
✅ Apply cross-validation and hyperparameter tuning
✅ Understand bias-variance tradeoff
✅ Work with ensemble methods

---

## Weekly Breakdown

### [Week 5: Supervised Learning - Regression](./Week_5_Regression/README.md)
**Time**: 15-18 hours

**Topics**:
- Linear regression from scratch
- Polynomial regression
- Ridge regression (L2 regularization)
- Lasso regression (L1 regularization)
- Elastic Net
- Evaluation metrics (MSE, RMSE, R², MAE)

**Code Examples**: 15+
**Datasets**: Housing prices, salary prediction, stock prices
**Project**: Predict house prices with 90%+ R²

---

### [Week 6: Supervised Learning - Classification](./Week_6_Classification/README.md)
**Time**: 15-18 hours

**Topics**:
- Logistic regression from scratch
- Decision trees (ID3, C4.5, CART)
- Random forests
- Support Vector Machines (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)

**Code Examples**: 20+
**Datasets**: Iris, Titanic, spam detection, credit card fraud
**Project**: Binary and multi-class classification with 90%+ accuracy

---

### [Week 7: Unsupervised Learning](./Week_7_Unsupervised/README.md)
**Time**: 12-15 hours

**Topics**:
- K-means clustering from scratch
- Hierarchical clustering
- DBSCAN
- PCA (Principal Component Analysis)
- t-SNE
- Anomaly detection

**Code Examples**: 12+
**Datasets**: Customer segmentation, image compression, outlier detection
**Project**: Customer segmentation with visualization

---

### [Week 8: Model Evaluation & Optimization](./Week_8_Evaluation/README.md)
**Time**: 12-15 hours

**Topics**:
- Train/validation/test splits
- K-fold cross-validation
- Stratified sampling
- Grid search and random search
- Confusion matrix analysis
- ROC curves and AUC
- Precision, recall, F1-score

**Code Examples**: 10+
**Project**: Complete ML pipeline with proper evaluation

---

### [Week 9: Advanced ML & Ensemble Methods](./Week_9_Advanced/README.md)
**Time**: 15-20 hours

**Topics**:
- Gradient boosting from scratch
- XGBoost deep dive
- LightGBM and CatBoost
- Stacking and blending
- Feature importance
- AutoML basics

**Code Examples**: 15+
**Datasets**: Kaggle-style competition data
**Project**: Ensemble model with 95%+ accuracy

---

## Key Algorithms Implemented from Scratch

### 1. Linear Regression
```python
class LinearRegression:
    """Linear Regression from scratch using gradient descent."""

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

### 2. Logistic Regression
```python
class LogisticRegression:
    """Logistic Regression from scratch."""

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return [1 if i > 0.5 else 0 for i in y_pred]
```

### 3. Decision Tree
```python
class DecisionTree:
    """Decision Tree Classifier from scratch."""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def entropy(self, y):
        """Calculate entropy."""
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def information_gain(self, parent, left_child, right_child):
        """Calculate information gain."""
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)

        gain = self.entropy(parent) - (
            weight_left * self.entropy(left_child) +
            weight_right * self.entropy(right_child)
        )
        return gain

    def best_split(self, X, y):
        """Find the best split."""
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = X[:, feature_idx] > threshold

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gain = self.information_gain(
                    y, y[left_idx], y[right_idx]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold
```

### 4. K-Means Clustering
```python
class KMeans:
    """K-Means Clustering from scratch."""

    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # Initialize centroids randomly
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            # Assign clusters
            clusters = self._assign_clusters(X)

            # Store old centroids
            old_centroids = self.centroids.copy()

            # Update centroids
            for i in range(self.n_clusters):
                if len(X[clusters == i]) > 0:
                    self.centroids[i] = X[clusters == i].mean(axis=0)

            # Check convergence
            if np.all(old_centroids == self.centroids):
                break

        return clusters

    def _assign_clusters(self, X):
        """Assign each point to nearest centroid."""
        distances = np.zeros((len(X), self.n_clusters))

        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)

        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)
```

---

## Projects

### Project 1: House Price Prediction
**Goal**: Predict house prices with R² > 0.90

**Tasks**:
- Load and explore dataset
- Feature engineering
- Handle missing values
- Implement linear regression
- Try polynomial features
- Apply regularization
- Evaluate model

**Deliverable**: Jupyter notebook with complete analysis

---

### Project 2: Credit Card Fraud Detection
**Goal**: Detect fraud with F1-score > 0.85

**Tasks**:
- Handle imbalanced dataset
- Feature scaling
- Implement logistic regression
- Try decision trees
- Ensemble methods
- ROC curve analysis

**Deliverable**: Production-ready fraud detector

---

### Project 3: Customer Segmentation
**Goal**: Segment customers into meaningful groups

**Tasks**:
- Exploratory data analysis
- Feature selection
- K-means clustering
- Determine optimal K
- Visualize clusters
- Profile each segment

**Deliverable**: Business insights report

---

### Project 4: Kaggle Competition
**Goal**: Top 25% on a Kaggle competition

**Tasks**:
- Complete ML pipeline
- Feature engineering
- Model selection
- Hyperparameter tuning
- Ensemble methods
- Submit predictions

**Deliverable**: Kaggle submission with writeup

---

## Tools & Libraries

### Core
- NumPy (for implementations)
- Pandas (for data handling)
- Matplotlib/Seaborn (for visualization)

### Validation
- Scikit-learn (for comparison only)
- XGBoost, LightGBM (Week 9)

---

## Assessment

### Skills Checklist
- [ ] Implement linear regression from scratch
- [ ] Implement logistic regression from scratch
- [ ] Build decision tree classifier
- [ ] Implement k-means clustering
- [ ] Understand bias-variance tradeoff
- [ ] Apply cross-validation properly
- [ ] Tune hyperparameters effectively
- [ ] Build complete ML pipelines
- [ ] Achieve 85%+ accuracy on projects

### Self-Assessment (1-5)
- Linear models: ___/5
- Tree-based models: ___/5
- Clustering: ___/5
- Evaluation: ___/5
- Feature engineering: ___/5

**Goal**: All areas 4+

---

## Key Concepts

### Bias-Variance Tradeoff
```
High Bias (Underfitting)
- Model too simple
- High training error
- High test error
→ Solution: More complex model

High Variance (Overfitting)
- Model too complex
- Low training error
- High test error
→ Solution: Regularization, more data

Sweet Spot
- Balanced complexity
- Low training error
- Low test error
```

### Regularization
```
L1 (Lasso): Sparse weights
L2 (Ridge): Small weights
Elastic Net: Both L1 and L2
```

### Cross-Validation
```
K-Fold: Split into K parts
Stratified: Preserve class distribution
Time Series: Respect temporal order
```

---

**Next**: [Module 3: Deep Learning](../Module_3_DeepLearning/README.md)

---

*Duration: 4-5 weeks*
*Code Examples: 72+*
*Projects: 4*
