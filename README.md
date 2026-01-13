# Telco Customer Churn Analysis
A Kaggle machine learning project for telecom customer churn prediction with business insights

This project analyzes customer churn in the telecommunications industry and builds machine learning models to identify customers at high risk of leaving. The goal is to support **proactive retention strategies** by maximizing the detection of potential churners.

---

## Business Objective

Customer churn is costly. Since contacting a customer is far cheaper than losing one, this project prioritizes **recall for churn customers** over overall accuracy, ensuring that as many at-risk customers as possible are identified.

---

## Dataset

- IBM Telco Customer Churn Dataset
- ~7,000 customers
- Target variable: `Churn` (Yes / No)
- Features include customer demographics, service usage, contract details, and billing information
- The dataset is **highly imbalanced** (~27% churn)

---

## Exploratory Data Analysis

Key findings:
- Churn is strongly associated with **short tenure** and **higher monthly charges**
- Customers in early contract stages are more likely to churn
- Several categorical features (e.g. contract type, tech support) show clear churn patterns
- Class imbalance motivates imbalance-aware modeling

---

## Modeling Approach

- Train/test split: 80% / 20% (stratified)
- Separate preprocessing pipelines for different model types:
  - **Logistic Regression**: one-hot encoding + standardization
  - **Tree-based models**: ordinal encoding, no scaling
- All preprocessing implemented via pipelines to avoid data leakage

---

## Models Evaluated

- Logistic Regression (Baseline)
- Logistic Regression + SMOTE
- Decision Tree
- Random Forest
- XGBoost

Models were evaluated using:
- Recall (Churn)
- F1-score (Churn)
- ROC-AUC

---

## Final Model Selection

**Logistic Regression (Baseline)** was selected as the final model because it:
- Achieved the **highest recall (78%)** for churn customers
- Delivered the **best ROC-AUC (0.833)**
- Is simple, interpretable, and production-friendly
- Outperformed more complex models for the business objective

---

## Business Insights

- The model identifies churn-prone customers effectively, accepting more false positives
- High-risk customers typically have:
  - Short tenure
  - High monthly charges
- Retention efforts should focus on **early-stage, high-cost customers**

---

## Visualization & Dashboards

Two Tableau-ready datasets are generated:
1. **Churn Profile Dashboard** – explores churn patterns using actual labels
2. **Predicted Churn Dashboard** – highlights high-risk customers using model predictions and probabilities

These dashboards enable actionable, data-driven retention decisions.

---

## Files
```{text}
├── telco.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── tableau_churn_profile.csv
├── tableau_predicted_churn.csv
```

---



## Tools & Libraries

- Python
- scikit-learn, imbalanced-learn
- XGBoost
- seaborn, matplotlib
- Tableau (for dashboarding)
