# Fraud-Detection
Project Title: Fraud Detection Pipeline — Supervised Learning on Imbalanced Transaction Data

# Enterprise Fraud Detection Pipeline (Supervised Learning)
### DecodeLabs Industrial Training Kit | Batch: 2026

---

## 📌 Project Overview
This project establishes a production-grade machine learning pipeline to detect fraudulent financial transactions within a heavily skewed and imbalanced environment. Instead of relying on traditional global accuracy—which is highly misleading in fraud analytics—this architecture prioritizes **Strict Precision**, **Recall (Sensitivity)**, and **ROC-AUC Score** to evaluate models.

The pipeline processes financial transactions,工程师 synthetic behavioral threat features (`Risk_Score`), applies **SMOTE** (Synthetic Minority Over-sampling Technique) to resolve class imbalance without data leakage, and optimizes algorithms using cross-validated hyperparameter tuning.

---

## 🏗️ Technical Pipeline Architecture

The workflow enforces a strict **Zero-Leakage Protocol**:
1. **Data Engineering:** Ingests raw transaction records and generates a composite continuous `Risk_Score` based on heuristic risk indicators (Order Cancellations, Payment Methods, and Volume anomalies).
2. **Feature Encoding & Scaling:** Categorical variables are cleaned, and high-variance numeric attributes are standardized via `StandardScaler`.
3. **Stratified Splitting:** Splitting the dataset ($80\%$ Train / $20\%$ Test) using stratified sampling to preserve the target distribution before resampling.
4. **Encapsulated SMOTE:** Synthetically populates sparse regions of the minority space. By executing this inside an `imblearn.pipeline.Pipeline`, synthetic vectors are restricted solely to the training folds.
5. **Hyperparameter Grid Search:** Evaluates model spaces across cross-validation paths to extract optimal parameter sets.

---

## 🛠️ Project Repository Structure

```text
├── Dataset for Data Analytics.csv     # Raw payment & transaction dataset
├── Data Science Project 2.pdf         # Corporate project handbook and guidelines
├── fraud_detection.py                 # Core production OOP/Functional pipeline script
├── README.md                          # Enterprise documentation (This file)
