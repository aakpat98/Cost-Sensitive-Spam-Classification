# Cost-Sensitive Spam Classification

This project implements a full machine learning pipeline for spam email classification using the [UCI Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase). It focuses on both maximizing predictive performance (Part A) and minimizing real-world misclassification costs (Part B).

---

## 📌 Project Overview

The classification task was approached in two parts:

### Part A: Accuracy-Optimized Modeling
- Compared multiple models (Logistic Regression, SVM, Random Forest, k-NN, Neural Network)
- Used **nested cross-validation** to select the best model based on **macro F1-score**
- Final evaluation included accuracy, F1-score, and AUC-ROC on a held-out test set

### Part B: Cost-Sensitive Classification
- Defined a custom cost function with a **10:1 penalty** on false positives (FP = 10, FN = 1)
- Repeated model selection using nested CV with the custom cost as the scoring metric
- Performed **manual tuning** of Random Forest to minimize total misclassification cost
- Achieved a **test cost of 222**, outperforming earlier configurations

---

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `Spambase-Workbook.ipynb` | Main Jupyter notebook with full data pipeline, modeling, tuning, and evaluation |
| `Spambase-Cost-Classification.pdf` | Final project report (markdown formatted with visual results) |
| `spambase.data` | Raw dataset from UCI |
| `spambase.names` | Feature names and attribute documentation |
| `spambase.DOCUMENTATION` | Additional metadata on the dataset (from UCI) |

---

## 🚀 Key Features

- Robust modeling using **pipelines, stratified splits, and class weighting**
- **Custom scoring** for cost-sensitive evaluation (using `make_scorer` in `GridSearchCV`)
- **Manual hyperparameter tuning** for improved business alignment
- Clear tracking of test performance and cost metrics with final evaluation reports

---

## 📊 Results Snapshot

| Model Type      | Metric Type        | Performance Summary                     |
|-----------------|--------------------|------------------------------------------|
| Random Forest   | Accuracy-focused   | F1 = 0.95, AUC ≈ 0.984                   |
| Random Forest   | Cost-sensitive     | Final test cost = 222, AUC ≈ 0.9852      |

---

## 💡 Business Impact

By aligning model optimization with business-defined error costs, this project demonstrates how data science can move beyond accuracy to deliver models that make smarter, risk-aware decisions.

---

## 📎 Dataset License

Data is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase), provided under open license for academic use.

---

## 🧠 Author

Project by [Your Name] – aspiring data scientist passionate about interpretable and impact-driven ML.
