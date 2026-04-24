# Heart Failure Prediction System

A full-stack machine learning web application that predicts heart failure risk from clinical data. Built with Flask, SQLAlchemy, and scikit-learn, featuring three role-based dashboards for patients, doctors, and administrators.

---

## Table of Contents

- [Dataset](#dataset)
- [Prediction Model Pipeline](#prediction-model-pipeline)
- [Flask Web Application](#flask-web-application)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Application](#running-the-application)

---

## Dataset

**Source:** Heart Failure Clinical Records Dataset (UCI / Kaggle)  
**Samples:** 299 patients  
**Target:** `DEATH_EVENT` — 0 (survived) or 1 (died during follow-up)

| Feature | Type | Description |
|---|---|---|
| age | Continuous | Age of the patient (years) |
| anaemia | Binary | Decrease in red blood cells (0/1) |
| creatinine_phosphokinase | Continuous | Level of CPK enzyme in blood (mcg/L) |
| diabetes | Binary | Patient has diabetes (0/1) |
| ejection_fraction | Continuous | Percentage of blood leaving the heart per beat |
| high_blood_pressure | Binary | Patient has hypertension (0/1) |
| platelets | Continuous | Platelet count in blood (kiloplatelets/mL) |
| serum_creatinine | Continuous | Level of creatinine in blood (mg/dL) |
| serum_sodium | Continuous | Level of sodium in blood (mEq/L) |
| sex | Binary | Biological sex — Male (1) / Female (0) |
| smoking | Binary | Patient smokes (0/1) |
| time | Continuous | Follow-up period in days |

---

## Prediction Model Pipeline

The full pipeline is documented in `notebook/heart_failure_ml.ipynb` and automated in `train_model.py`.

### Step 1 — Library Imports

All necessary libraries are imported at the top of the notebook:

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ...
```

### Step 2 — Data Loading

The CSV is loaded with `pandas.read_csv()`. Basic inspection is performed:

```python
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.shape      # (299, 13)
df.dtypes
df.info()
df.describe()
```

### Step 3 — Exploratory Data Analysis (EDA)

Several visualisations are produced to understand the data:

- **Class distribution** — bar chart and pie chart of `DEATH_EVENT` (203 survived, 96 died; ~2.1:1 imbalance)
- **Feature distributions** — overlapping histograms of each feature split by outcome (survived vs. died), showing which features separate the classes
- **Box plots** — per-feature box plots grouped by `DEATH_EVENT` to highlight median shifts and outliers
- **Correlation heatmap** — lower-triangular heatmap using seaborn; `time`, `serum_creatinine`, and `ejection_fraction` have the strongest correlations with the target

Key observations from EDA:
- Patients who died had notably lower `ejection_fraction` and `serum_sodium`
- Higher `serum_creatinine` is associated with death events
- `time` (follow-up period) shows the strongest single-feature signal

### Step 4 — Preprocessing

```python
FEATURES = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'high_blood_pressure', 'platelets',
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

X = df[FEATURES].values
y = df['DEATH_EVENT'].values
```

The dataset has **no missing values**, so no imputation is required.

**Train/test split** — 80% training, 20% test, stratified on the target to preserve the class ratio:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Result: 239 train samples, 60 test samples
```

**Standard Scaling** — `StandardScaler` is fit on training data only and applied to both splits:

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

This is critical for SVM, which is sensitive to feature magnitude.

### Step 5 — Model Training

Three classifiers are trained on the scaled training data:

**Model 1 — Support Vector Machine (SVM)**
```python
SVC(kernel='rbf', probability=True, random_state=42)
```
Uses the RBF (Gaussian) kernel. `probability=True` enables `predict_proba()` for risk scoring. SVM finds the maximum-margin hyperplane in the feature space.

**Model 2 — Random Forest**
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```
An ensemble of 100 decision trees trained on random feature subsets (bagging). Each tree votes; the majority class wins. Naturally provides feature importances.

**Model 3 — Gradient Boosting**
```python
GradientBoostingClassifier(n_estimators=100, random_state=42)
```
Builds trees sequentially, each one correcting the residual errors of the previous. Generally strong on tabular data.

### Step 6 — Evaluation

Each model is evaluated on the held-out test set using:

| Metric | Formula | Why it matters |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Precision | TP / (TP+FP) | How many predicted positives are real |
| Recall | TP / (TP+FN) | How many actual positives are caught |
| F1-Score | 2*(P*R)/(P+R) | Balance of precision and recall |
| ROC-AUC | Area under ROC curve | Discrimination ability across thresholds |
| CV Accuracy | 5-fold cross-validation mean | Stability across data splits |

Results on the test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM (RBF) | 76.7% | 72.7% | 42.1% | 53.3% | 0.845 |
| Random Forest | 83.3% | 80.0% | 63.2% | 70.6% | 0.891 |
| Gradient Boosting | 83.3% | 80.0% | 63.2% | 70.6% | 0.845 |

Visualisations produced at this step:
- **Confusion matrices** — side-by-side for all three models
- **ROC curves** — all three plotted together with AUC in the legend
- **Grouped bar chart** — all six metrics compared across models

### Step 7 — Best Model Selection

Random Forest is selected as the best model based on accuracy (83.3%) and the highest ROC-AUC (0.891), indicating the best overall discrimination ability.

```python
best_name = max(evaluation, key=lambda n: evaluation[n]['accuracy'])
# Result: 'Random Forest'
```

### Step 8 — Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation is applied to the best model:

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                   param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_sc, y_train)
```

Best parameters found: `{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}`  
Tuned test accuracy: **85.0%** (improvement from 83.3%)

### Step 9 — Feature Importance

The tuned Random Forest exposes feature importances derived from mean impurity decrease across all trees:

| Rank | Feature | Importance |
|---|---|---|
| 1 | time | 0.4002 |
| 2 | serum_creatinine | 0.1603 |
| 3 | ejection_fraction | 0.1298 |
| 4 | creatinine_phosphokinase | 0.0742 |
| 5 | age | 0.0663 |

`time` (follow-up period) is the dominant predictor, consistent with clinical literature showing survival time as a proxy for disease severity.

### Step 10 — Save Model Artifacts

```python
import pickle

with open('app/ml/best_model.pkl', 'wb') as f: pickle.dump(final_model, f)
with open('app/ml/scaler.pkl',     'wb') as f: pickle.dump(scaler, f)
with open('app/ml/feature_names.pkl', 'wb'): pickle.dump(FEATURES, f)
```

These three files are loaded at runtime by the Flask application to serve predictions.

---

## Flask Web Application

The web app is built with **Flask**, **Flask-SQLAlchemy** (SQLite), and **Flask-Login**, with **Chart.js** for interactive visualisations.

### Three Role-Based Dashboards

**Patient**
- Submit clinical data through a 12-field form to receive an instant risk prediction
- View prediction history with a risk trend line chart and risk-level doughnut chart
- See doctor notes attached to each prediction

**Doctor**
- View all assigned patients with their latest risk level
- Interactive charts: patient risk distribution, monthly prediction activity, per-patient risk bar chart
- Add or edit clinical notes on any patient prediction via a modal

**Admin**
- System-wide statistics: total users, predictions, high-risk count, unassigned patients
- Four live charts: user role distribution, prediction risk breakdown, prediction trend over time, user registration trend
- Full user management: create users, change roles, assign patients to doctors, delete accounts

### Database Models

- `User` — stores credentials, role (patient / doctor / admin), and creation timestamp
- `Prediction` — stores all 12 clinical features, model output (result + probability + risk level), doctor notes, and reviewer ID
- `DoctorPatient` — many-to-many assignment table linking doctors to patients

---

## Project Structure

```
heart_failure/
├── notebook/
│   └── heart_failure_ml.ipynb     # Full prediction pipeline notebook
├── app/
│   ├── __init__.py                # App factory, extensions
│   ├── models.py                  # SQLAlchemy models
│   ├── routes/
│   │   ├── auth.py                # Login, register, logout
│   │   ├── patient.py             # Patient dashboard and prediction
│   │   ├── doctor.py              # Doctor dashboard and notes
│   │   └── admin.py               # Admin dashboard and user management
│   ├── templates/
│   │   ├── base.html              # Shared sidebar layout
│   │   ├── auth/                  # Login and register pages
│   │   ├── patient/               # Patient dashboard, predict form, result, profile
│   │   ├── doctor/                # Doctor dashboard, patient detail
│   │   └── admin/                 # Admin dashboard, user management
│   ├── static/
│   │   └── css/style.css          # Application stylesheet
│   └── ml/                        # Saved model artifacts (generated by train_model.py)
├── heart_failure_clinical_records_dataset.csv
├── train_model.py                 # Standalone model training script
├── run.py                         # Application entry point
├── config.py                      # Flask configuration
└── requirements.txt
```

---

## Setup and Installation

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone https://github.com/GNair06/heart_failure.git
cd heart_failure

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model and save artifacts
python train_model.py
```

---

## Running the Application

```bash
python run.py
```

Open `http://localhost:5001` in your browser.

**Default admin account** (created automatically on first run):
- Email: `admin@heartcare.com`
- Password: `admin123`

**Workflow:**
1. Log in as admin and go to **Manage Users** to create patient and doctor accounts
2. Assign patients to doctors using the Assign Doctor dropdown
3. Log in as a patient and submit clinical data to get a prediction
4. Log in as a doctor to view assigned patients, risk charts, and add notes
