# Bank Marketing — Random Forest (Calibrated)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

Predicting Term-Deposit Subscription from the UCI “Bank Marketing” dataset

---

## 📌 Executive Summary

This project develops a **calibrated Random Forest classifier** to predict whether a client will subscribe to a **term deposit** following a direct marketing campaign. The workflow emphasizes:

* **Feature engineering** for marketing history & socio-economic signals
* **Stratified cross-validation** with random/grid search tuning
* **Probability calibration** for trustworthy decision thresholds

**Key Outcomes (Test Split 25%):**

* **Accuracy:** 0.851
* **ROC-AUC:** 0.817
* **PR-AUC (Avg Precision):** 0.496
* **Confusion Matrix:** `[[8014, 1123], [413, 747]]`
* **Positive class (subscribe)**: Precision 0.40 | Recall 0.64 | F1 0.49

> ⚡ In a highly imbalanced setting (\~11.3% positives), **recall prioritization** ensures more true subscribers are identified, which is crucial for campaign ROI.

---

## 📊 Dataset Overview

* **Source:** UCI Bank Marketing Dataset
* **Shape:** 41,188 rows × 21 columns
* **Target:** `y` (converted into binary `target`)
* **Class Balance:** 11.3% `yes`, 88.7% `no`

**Target Distribution:**
![Target Count](./images/Target%20count.png)

**Job vs Target:**
![Job vs Target](./images/job%20vs%20target.png)

**Campaign count vs Target:**
![Campaign Count](./images/campaing%20count%20by%20target.png)

**Euribor 3-month rate vs Target:**
![Euribor](./images/euribor%203%20month%20rate%20density%20vs%20target.png)

> 🔎 The dataset contains “unknown” values in `default`, `education`, `housing`, `loan`, etc. The `duration` variable was **removed** to avoid data leakage.

---

## 🛠️ Approach & Pipeline

### 🔹 Feature Engineering

* Dropped `duration` (leakage)
* Created marketing history indicators:

  * `was_prev_contacted` (binary)
  * `pdays_when_contacted` (numeric, NaN for never contacted)
* Engineered binary target variable

### 🔹 Preprocessing

* **Categorical:** OneHotEncoder
* **Numerical:** Pass-through

### 🔹 Model

* `RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)`
* Integrated in **Pipeline** with preprocessing

### 🔹 Train/Test Split

* Train: 30,891 rows
* Test: 10,297 rows

---

## ⚙️ Hyperparameter Tuning & Calibration

* RandomizedSearchCV (12k subsample) + GridSearchCV
* Optimized for **ROC-AUC**
* Final model wrapped in **CalibratedClassifierCV (isotonic, cv=5)**

**Performance after Calibration:**

* ROC-AUC: 0.817
* PR-AUC: 0.496
* Brier Score: 0.074

**ROC Curve:**
![ROC Curve](./images/R0o%20curve.png)

---

## ✅ Results (Test Set)

**Classification Report:**

```
Class 0: precision 0.95 | recall 0.88 | f1 0.91 | support 9137
Class 1: precision 0.40 | recall 0.64 | f1 0.49 | support 1160
Overall accuracy: 0.851 | ROC-AUC: 0.817 | PR-AUC: 0.496
```

**Confusion Matrix:**
![Confusion Matrix](./images/confusin%20metriucs.png)

---

## 🔑 Feature Importance & Explainability

**Top Global Feature Importances:**

1. `nr.employed`
2. `euribor3m`
3. `emp.var.rate`
4. `cons.conf.idx`
5. `age`
6. `pdays_when_contacted`
7. `cons.price.idx`
8. `campaign`
9. `month_may`, `month_oct`
10. `was_prev_contacted`

![Feature Importances](./images/Feature%20inmportances.png)

**SHAP Analysis:**

* Global importance:
  ![SHAP Importance](./images/shap%20va;ues%20importances.png)
* Impact distribution:
  ![SHAP Distribution](./images/shap%20va;ue%20prediction%20imoact.png)
* Individual predictions:
  ![SHAP Impact](./images/shap%20values%20impact%20on%20output.png)

---

## 📈 Business Framing & Impact

* **Lead scoring**: prioritize clients most likely to subscribe
* **Campaign efficiency**: maximize ROI by targeting likely converters
* **Imbalance-aware**: recall-oriented thresholding captures more subscribers
* **Explainability**: SHAP-based insights enhance trust and compliance

---

## ⚡ How to Reproduce

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U scikit-learn==1.5.1 pandas numpy matplotlib seaborn joblib
```

### Run

1. Place `bank-additional-full.csv` in `./data/`
2. Run notebook → trains, tunes, and calibrates the model
3. Artifacts are saved in `./models` and `./outputs`

---

## 📂 Project Structure

```
.
├── notebooks/
│   └── Random forest tree banking.ipynb
├── models/
│   ├── rf_bank_marketing_calibrated.joblib
│   └── encoded_feature_names.csv
├── outputs/
│   └── feature_importances_top25.csv
├── images/
│   ├── Target count.png
│   ├── job vs target.png
│   ├── campaing count by target.png
│   ├── euribor 3 month rate density vs target.png
│   ├── R0o curve.png
│   ├── confusin metriucs.png
│   ├── Feature inmportances.png
│   ├── shap va;ues importances.png
│   ├── shap va;ue prediction imoact.png
│   ├── shap values impact on output.png
├── data/
│   └── bank-additional-full.csv  (not tracked)
├── README.md
└── requirements.txt
```

---

## ⚖️ Governance, Risks & Next Steps

* **Fairness:** Age & employment-related features → monitor bias
* **Concept drift:** Macro variables → retrain regularly
* **Data quality:** Handle “unknown” values in defaults/education
* **Future Work:** Gradient boosting, cost-sensitive learning, real-time REST API

---

## 📌 Attribution

* **Dataset:** UCI Bank Marketing
* **Author:** Steffin Thomas
* **Environment:** Python 3.12.7, scikit-learn 1.5.1
* **Notebook Run:** Aug 2025

---

## 🚀 Hire-Ready Highlights

* Clean & reproducible **end-to-end ML pipeline**
* Balanced handling of **class imbalance & calibration**
* Strong **business framing + explainability** for recruiter review
* Deployment-ready artifacts

---
