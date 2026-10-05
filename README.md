# 📊 Inovegen Internship — Task 3: Customer Churn Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![Inovegen](https://img.shields.io/badge/Inovegen-AI%2FML%20Internship-purple.svg)](https://inovegen.com)

---

## 📌 Executive Overview
Customer churn—the rate at which customers discontinue their subscriptions—directly impacts company profitability, as acquiring a new customer costs **5× to 7× more** than retaining an existing one. 

This project fulfills **Inovegen AI/ML Internship Task 3**, implementing an end-to-end Machine Learning pipeline that predicts customer churn on tabular telecom data. It encompasses diagnostic data cleaning, exploratory data analysis (EDA), stratified feature engineering, and a benchmark comparison between **Logistic Regression** and **Random Forest Classifier**.

---

## 🗂️ Dataset Source & Reference
- **Dataset:** Telco Customer Churn Dataset (IBM / Kaggle benchmark)
- **Observations:** `7,043` original customer records across `21` attributes
- **Target Variable:** `Churn` (`Yes` = 1, `No` = 0)
- **Public URL Reference:** [IBM Telco Customer Churn on ICP4D](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

---

## 🧹 Data Cleaning & Preprocessing Pipeline
1. **Unsuitable Identifier Removal:** Dropped `customerID` (7,043 unique alphanumeric strings) to prevent the models from memorizing arbitrary identifiers and overfitting.
2. **Duplicate Remediation:** Detected and eliminated 22 duplicate rows, preserving 7,021 unique, pristine customer profiles.
3. **Implicit Missing Value Imputation (`TotalCharges`):**
   - Discovered 11 whitespace strings (`" "`) causing Pandas to misinterpret numerical charges as an `object` (string) column.
   - All 11 records had `tenure == 0` (brand new subscribers).
   - Rather than introducing synthetic bias via mean/median imputation, these were imputed with `0.0`—the physical ground truth of zero accumulated charges.
4. **Feature Encoding:**
   - Target `Churn` mapped to binary integers (`Yes` $\rightarrow 1$, `No` $\rightarrow 0$).
   - One-Hot Encoded categorical variables using `pd.get_dummies(..., drop_first=True)` to prevent the dummy variable trap (multicollinearity), yielding **30 numeric features**.
5. **Stratified Train-Test Split:**
   - 80% Training Set (`5,616` records)
   - 20% Holdout Test Set (`1,405` records)
   - Stratified on `y` (`stratify=y`) to maintain the identical 26.6% churn ratio across both subsets.
6. **Feature Standardization:** Applied `StandardScaler` to continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`), fitting strictly on the training set to prevent data leakage.

---

## 📈 Exploratory Data Analysis (Key Visual Insights)

1. **Target Class Imbalance:**
   - Retained: **73.4%** (`5,153` customers) | Churned: **26.6%** (`1,868` customers).
   - *Implication:* Demonstrates why raw Accuracy is deceptive (a dummy model guessing "Stay" would achieve 73.4% accuracy while catching zero churners), establishing the necessity of Recall, F1-Score, and ROC-AUC.
2. **Contract Structure (15× Risk Multiplier):**
   - **Month-to-month contracts:** **42.2% churn rate**
   - **One-year contracts:** **11.3% churn rate**
   - **Two-year contracts:** **2.8% churn rate**
   - *Implication:* Lack of contractual lock-in is the single largest structural vulnerability in customer retention.
3. **Customer Lifecycle & Billing Vulnerabilities:**
   - Churn density peaks sharply during the **first 0 to 10 months** of tenure.
   - Churn heavily concentrates among customers with monthly fees between **$70 and $90/month** (primarily Fiber Optic users without bundled support).

---

## 🤖 Model Benchmark & Evaluation

We trained and evaluated two complementary architectures on the 20% unseen test set (`1,405` customers):

| Evaluation Metric | Logistic Regression (Champion) | Random Forest Classifier |
| :--- | :---: | :---: |
| **Accuracy** | **80.21%** | 79.72% |
| **Precision** | 65.99% | **66.17%** |
| **Recall (Sensitivity)** | **52.15%** | 47.85% |
| **F1-Score** | **58.26%** | 55.54% |
| **ROC-AUC Score** | **84.04%** | 83.74% |
| **Successfully Caught Churners (TP)** | **194 / 372** | 178 / 372 |
| **Missed Churners (FN)** | **178** | 194 |

### 🏆 Champion Model Selection Rationale:
**Logistic Regression** is selected as the operational champion. In subscription churn, **Recall** is the most critical metric because a False Negative (a missed churner who cancels quietly) results in direct recurring revenue loss. Logistic Regression captured **16 additional churners** over Random Forest while achieving a superior ROC-AUC of **84.04%**.

---

## 🔍 Feature Importance (Top Churn Drivers)
Extracted Gini feature importances from the Random Forest model revealed that **nearly 45% of customer decisions are governed by three core lifecycle and financial factors**:

1. **`tenure` (20.5% importance):** Customer lifespan with the company.
2. **`TotalCharges` (13.8% importance):** Cumulative financial commitment.
3. **`MonthlyCharges` (10.0% importance):** Immediate monthly price burden.
4. **`Contract_Two year` (6.2% importance):** Strongest protective retention barrier.
5. **`InternetService_Fiber optic` (5.1% importance):** High price/service dissatisfaction vector.

---

## 💡 Strategic Business Recommendations

1. **Onboarding Safety Net (Months 1–6):** Focus customer success efforts, tutorials, and automated check-ins on new subscribers during their first 90 days to guide them safely past the high-risk 0–10 month window.
2. **Contract Migration Incentives:** Design proactive campaigns offering a small discount (e.g., $5 off/month) or free speed tiers to transition month-to-month subscribers into 1- or 2-year commitments.
3. **Support Bundling for Premium Fiber:** Bundle `TechSupport` and `OnlineSecurity` into $70–$90/month Fiber Optic plans, as customers with active technical support exhibit substantially lower churn.

---

## 🚀 How to Reproduce
1. Open the notebook in Google Colab or Jupyter:
   ```bash
   jupyter notebook Inovegen_Task3_Customer_Churn_Prediction.ipynb
