# 🏦 Bank Customer Churn Prediction using Machine Learning

This repository contains the implementation of a machine learning pipeline to predict customer churn in the banking industry. The aim is to help banks retain customers by identifying those likely to exit using historical data and predictive modeling techniques.

## 📌 Project Overview

Customer churn is a major challenge in the banking sector. Retaining customers is significantly more cost-effective than acquiring new ones. This project builds and evaluates multiple machine learning models to classify whether a customer is likely to churn.

## 📊 Dataset

- **Source**: [Kaggle - Predicting Churn for Bank Customers](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers)
- **Size**: 10,000 records × 14 features
- **Target**: `Exited` (1 = Churned, 0 = Stayed)

### Features include:

- Customer demographics (Age, Gender, Geography)
- Banking activity (Tenure, Balance, NumOfProducts)
- Indicators (HasCrCard, IsActiveMember)
- Financial metrics (CreditScore, EstimatedSalary)

## 🔧 Preprocessing Steps

- Removed non-informative columns (`RowNumber`, `CustomerId`, `Surname`)
- Label encoding for `Gender`
- One-hot encoding for `Geography`
- Standard scaling for features
- Data split: 80% training, 20% testing

## 📈 Exploratory Data Analysis

- Pie chart of churn distribution
- Count plots for categorical variables
- Boxplots and correlation matrix for numerical features
- Feature-target correlation analysis

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

## 🏁 Model Evaluation

Evaluated using:

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation
- Hyperparameter tuning via Grid Search and Randomized Search

### ✅ Best Performing Model:
**Random Forest Classifier**
- Accuracy: 86.35%
- F1 Score: 0.60 (Class 1)
- Balanced Precision & Recall

## 🛠️ Libraries Used

```python
pandas, numpy, matplotlib, seaborn
sklearn: preprocessing, model_selection, metrics, ensemble, svm, tree, linear_model
joblib
