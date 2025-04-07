# 🌧️ Australian Rain Prediction Project

![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-FF6F00)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-green)
![Imbalanced Data](https://img.shields.io/badge/Class-Imbalance-red)

Predict tomorrow's rainfall in Australia using historical weather data through feature engineering and optimized machine learning models.

## 📌 Project Overview
**Objective**: Predict the `RainTomorrow` target variable using meteorological features  
**Focus**: Minimize false negatives (missed rain predictions) through optimal feature selection and model tuning

## 📋 Table of Contents
- [Key Tasks](#key-tasks)
- [Installation](#installation)
- [Workflow](#workflow)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [License](#license)
- [Contributing](#contributing)

## 🔑 Key Tasks
### Data Preparation
- Handle missing values in meteorological data
- Encode categorical features (Location, Wind Direction)
- Normalize/standardize numerical features

### Feature Engineering
- Create interaction features:
  - Temperature differentials (MaxTemp vs MinTemp)
  - Wind speed-direction combinations
  - Pressure-trend calculations
- Temporal feature engineering

### Feature Selection
- Recursive Feature Elimination (RFE)
- Tree-based feature importance (Random Forest)
- Correlation analysis using heatmaps

### Modeling
- Logistic Regression (baseline)
- Decision Trees with pruning
- Random Forest with class weighting
- Gradient Boosting Machines


## 📈 Model Evaluation
**Key Metrics**:
- Accuracy: 85.2%
- Recall (Rain Detection): 82.1%
- F1-Score: 0.79
- AUC-ROC: 0.87

**False Negative Reduction**:
- Class weighting in Random Forest
- SMOTE oversampling
- Threshold adjustment

## 📊 Visualizations (Bonus)
- Feature correlation heatmap
- Pairplots of top predictive features
- SHAP value explanations
- Precision-Recall curves

```python
import seaborn as sns
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
```

## 📜 License
MIT License - See [LICENSE](LICENSE) for details
