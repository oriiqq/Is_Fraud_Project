# Credit Card Fraud Detection
## Overview

This project focuses on detecting credit card fraud using machine learning, following a systematic data science workflow:

Flat File Creation → Data Cleansing → Exploratory Data Analysis (EDA) → Feature Engineering & Modeling.

The goal is to identify fraudulent transactions with high precision and interpretability while maintaining both academic rigor and practical relevance.

## Motivation

Credit card fraud is a critical challenge for global financial systems. Fraudulent transactions are rare compared to legitimate ones, making it difficult for standard classification algorithms to detect them effectively.

This project is motivated by the need for a transparent, reproducible workflow to analyze, prepare, and model credit card transaction data. Emphasis is placed on handling class imbalance, ensuring interpretability, and maintaining data quality.

## Dataset

The dataset consists of anonymized credit card transactions. Each row represents a single transaction labeled as legitimate (0) or fraudulent (1). Features are numeric and include PCA-transformed components, transaction amount, and time.

## Key points:

Features: 30 numerical attributes (V1–V28, plus Amount and Time)

Target: Binary fraud indicator (0 or 1)

Challenge: Extreme imbalance — typically <0.2% of transactions are fraudulent

## Project Workflow

The project is organized into four primary notebooks:

1. Data_Preparation.ipynb — Flat File Creation

Purpose: Assemble a unified, analysis-ready dataset from raw fragments.
Main actions:

Load raw or distributed files

Merge multiple sources into a single flat structure

Standardize column names, formats, and encodings

Verify transaction identifiers and timestamps

Save the final clean flat file (credit_card_flat.csv)

Motivation: Ensures consistency and reproducibility for downstream analysis.
Challenges: Inconsistent schemas and missing identifiers required careful validation.

2. Data_Cleansing.ipynb

Purpose: Clean and verify dataset integrity.
Main actions:

Check for duplicates and missing values

Validate data types and correct misformatted entries

Detect and handle outliers (IQR or z-score)

Confirm logical consistency (e.g., transaction time and amount)

Motivation: Reliable models require high-quality input data.
Challenges: Careful validation was needed to avoid removing legitimate transactions that resemble fraud.

3. EDA.ipynb — Exploratory Data Analysis

Purpose: Explore and visualize trends, correlations, and potential fraud predictors.
Main actions:

Visualize transaction amount and time distributions

Highlight class imbalance

Examine PCA features for fraud separation

Identify patterns distinguishing legitimate from fraudulent transactions

Insights:

Fraudulent transactions often have lower amounts and occur at distinct times

Certain PCA components provide strong class separability

Extreme class imbalance requires special evaluation metrics

Challenges: Rare fraud events required log-scaling and careful sampling to avoid misleading patterns.

4. Eng_Selc_Eval.ipynb — Feature Engineering, Selection & Modeling

Purpose: Engineer features, train models, and evaluate performance.
Main actions:

Feature selection using correlation and model-based importance

Train multiple algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost

Evaluate with metrics suitable for imbalanced datasets (precision, recall, F1-score, ROC-AUC)

Compare models and visualize results

Results:

Best models: Random Forest and XGBoost achieved the highest recall and ROC-AUC

Important features: V4, V12, V17

Logistic Regression provided baseline interpretability

## Challenges: Severe class imbalance required focusing on recall and ROC-AUC rather than raw accuracy. Ensemble models required careful hyperparameter tuning.

## Methodology Summary
Step	Description	Key Libraries
Data Preparation	Combine and clean raw sources into one dataset	pandas, numpy
Data Cleansing	Remove inconsistencies and verify integrity	pandas, numpy
EDA	Visual exploration and trend detection	matplotlib, seaborn
Modeling	Train and evaluate machine learning models	scikit-learn, xgboost
