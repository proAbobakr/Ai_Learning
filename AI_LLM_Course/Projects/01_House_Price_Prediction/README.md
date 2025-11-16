# Capstone Project 1: House Price Prediction

## Project Overview

Build a complete machine learning pipeline to predict house prices with R² > 0.90.

**Duration**: 8-12 hours
**Difficulty**: Intermediate
**Skills**: Regression, Feature Engineering, Model Evaluation

---

## Objectives

✅ Load and explore real estate dataset
✅ Perform exploratory data analysis (EDA)
✅ Handle missing values and outliers
✅ Engineer meaningful features
✅ Implement multiple regression models
✅ Compare model performance
✅ Achieve R² score > 0.90
✅ Create visualizations and report

---

## Dataset

We'll use a synthetic house prices dataset with the following features:

### Features
1. **SquareFeet**: Living area in square feet
2. **Bedrooms**: Number of bedrooms
3. **Bathrooms**: Number of bathrooms
4. **Age**: Age of the house in years
5. **LotSize**: Lot size in square feet
6. **Garage**: Number of garage spaces
7. **Neighborhood**: Categorical (A, B, C)
8. **Condition**: House condition (1-5)
9. **YearBuilt**: Year the house was built
10. **HasPool**: Binary (0 or 1)

### Target
- **Price**: House price in dollars

---

## Project Structure

```
01_House_Price_Prediction/
├── README.md (this file)
├── data/
│   ├── house_prices.csv
│   └── data_description.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_final_model.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── models.py
│   └── evaluator.py
├── results/
│   ├── plots/
│   └── model_comparison.csv
└── report.md
```

---

## Step-by-Step Guide

### Step 1: Data Exploration (2 hours)

**Tasks**:
- Load the dataset
- Check for missing values
- Analyze distributions
- Identify outliers
- Understand correlations

**Code Template**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/house_prices.csv')

# Basic info
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

**Deliverable**: `01_data_exploration.ipynb` with complete EDA

---

### Step 2: Data Preprocessing (2 hours)

**Tasks**:
- Handle missing values
- Remove or cap outliers
- Encode categorical variables
- Split train/test sets
- Scale features

**Code Template**:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class HousePricePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(drop='first', sparse=False)

    def handle_missing(self, df):
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Fill categorical with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def remove_outliers(self, df, columns, n_std=3):
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] >= mean - n_std*std) &
                   (df[col] <= mean + n_std*std)]
        return df

    def fit_transform(self, X_train):
        # Separate numeric and categorical
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(include=['object']).columns

        # Scale numeric
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])

        # Encode categorical
        if len(cat_cols) > 0:
            encoded = self.encoder.fit_transform(X_train[cat_cols])
            # TODO: Combine with numeric features

        return X_train
```

---

### Step 3: Feature Engineering (2 hours)

**Tasks**:
- Create interaction features
- Polynomial features
- Domain-specific features
- Feature selection

**Example Features**:
```python
def engineer_features(df):
    # Create new features
    df['PricePerSqFt'] = df['SquareFeet'] / df['Price']  # For analysis
    df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']
    df['AgeSquared'] = df['Age'] ** 2
    df['SqFt_Bedrooms'] = df['SquareFeet'] * df['Bedrooms']
    df['HasBasement'] = (df['BasementSqFt'] > 0).astype(int)
    df['LuxuryScore'] = (
        df['Condition'] * 0.3 +
        df['HasPool'] * 0.4 +
        (df['Garage'] > 1) * 0.3
    )

    # Binning
    df['AgeCategory'] = pd.cut(df['Age'],
                                bins=[0, 10, 30, 50, 100],
                                labels=['New', 'Recent', 'Old', 'VeryOld'])

    return df
```

---

### Step 4: Model Training (3 hours)

**Tasks**:
- Implement linear regression
- Try polynomial regression
- Apply regularization (Ridge, Lasso)
- Ensemble methods
- Cross-validation

**Models to Implement**:

```python
# 1. Simple Linear Regression
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 2. Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# 3. Ridge Regression
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# 4. Lasso Regression
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# 5. Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
```

---

### Step 5: Model Evaluation (2 hours)

**Tasks**:
- Calculate metrics (R², RMSE, MAE)
- Create prediction plots
- Residual analysis
- Feature importance
- Model comparison

**Evaluation Code**:
```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f\"{model_name} Results:\")
    print(f\"  R² Score: {r2:.4f}\")
    print(f\"  RMSE: ${rmse:,.2f}\")
    print(f\"  MAE: ${mae:,.2f}\")

    # Prediction vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'{model_name}: Predictions vs Actual\\nR² = {r2:.4f}')
    plt.grid(True)
    plt.show()

    return {'r2': r2, 'rmse': rmse, 'mae': mae}
```

---

### Step 6: Final Model and Report (1-2 hours)

**Tasks**:
- Select best model
- Fine-tune hyperparameters
- Create final predictions
- Write comprehensive report

---

## Evaluation Criteria

### Technical (70 points)
- [ ] R² Score > 0.90 (20 points)
- [ ] Proper data preprocessing (10 points)
- [ ] Feature engineering creativity (15 points)
- [ ] Multiple models compared (10 points)
- [ ] Cross-validation used (10 points)
- [ ] Code quality and documentation (5 points)

### Report (30 points)
- [ ] Clear problem statement (5 points)
- [ ] EDA insights (5 points)
- [ ] Methodology explanation (10 points)
- [ ] Results visualization (5 points)
- [ ] Conclusions and learnings (5 points)

**Total: 100 points**

---

## Bonus Challenges (+20 points)

1. **Hyperparameter Tuning** (+5 points)
   - Use Grid Search or Random Search
   - Document the tuning process

2. **Feature Selection** (+5 points)
   - Implement feature selection algorithms
   - Compare performance with/without selection

3. **Ensemble Model** (+5 points)
   - Create a stacking or blending ensemble
   - Beat individual model performance

4. **Interactive Dashboard** (+5 points)
   - Create a Streamlit/Gradio app
   - Allow users to input features and get predictions

---

## Expected Results

### Minimum Requirements
- R² Score: > 0.90
- RMSE: < $30,000
- MAE: < $20,000

### Excellent Performance
- R² Score: > 0.95
- RMSE: < $20,000
- MAE: < $15,000

---

## Resources

### Documentation
- Scikit-learn Regression: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/

### Tutorials
- Feature Engineering: https://www.kaggle.com/learn/feature-engineering
- Model Tuning: https://scikit-learn.org/stable/modules/grid_search.html

---

## Submission

Submit the following:
1. Complete Jupyter notebooks (all 4)
2. Source code (all .py files)
3. Results (plots and comparison CSV)
4. Final report (report.md)
5. Trained model (pickle or joblib file)

---

## Timeline

**Week 1** (8-12 hours):
- Day 1-2: Data exploration and preprocessing
- Day 3: Feature engineering
- Day 4-5: Model training and evaluation
- Day 6: Final model and report
- Day 7: Review and polish

---

## Tips for Success

1. **Start Simple**: Begin with basic linear regression, then add complexity
2. **Visualize Everything**: Plots reveal insights numbers can't
3. **Document As You Go**: Don't wait until the end
4. **Experiment**: Try unusual feature combinations
5. **Cross-Validate**: Don't trust a single train/test split
6. **Ask for Help**: Use forums, Discord, Stack Overflow

---

## Next Projects

After completing this:
- **Project 2**: Credit Card Fraud Detection (Classification)
- **Project 3**: Customer Segmentation (Clustering)
- **Project 4**: Kaggle Competition

---

**Good luck! 🚀**

*Estimated completion time: 8-12 hours*
*Difficulty: Intermediate*
*Required skills: Python, NumPy, Pandas, Scikit-learn*
