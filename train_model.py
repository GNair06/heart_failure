"""
Heart Failure Prediction — Model Training Script
Run: python train_model.py
Trains SVM, Random Forest, and Gradient Boosting classifiers on the dataset,
evaluates all three, picks the best by accuracy, and saves it + the scaler.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)

warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'heart_failure_clinical_records_dataset.csv')
ML_DIR    = os.path.join(BASE_DIR, 'app', 'ml')
os.makedirs(ML_DIR, exist_ok=True)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("HEART FAILURE PREDICTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nClass distribution:\n{df['DEATH_EVENT'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ─── 2. Feature / Target Split ────────────────────────────────────────────────
FEATURES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
]

X = df[FEATURES].values
y = df['DEATH_EVENT'].values

print(f"\nFeatures used ({len(FEATURES)}): {FEATURES}")
print(f"Target: DEATH_EVENT  |  0=Survived  1=Died")

# ─── 3. Train / Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")

# ─── 4. Scaling ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 5. Define Models ─────────────────────────────────────────────────────────
models = {
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ─── 6. Train & Evaluate All Models ──────────────────────────────────────────
print("\n" + "─" * 60)
print("MODEL TRAINING & EVALUATION")
print("─" * 60)

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cv   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy').mean()

    results[name] = {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'cv_accuracy': cv,
    }

    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  CV Accuracy: {cv:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))

# ─── 7. Comparison Summary ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("COMPARISON SUMMARY")
print("─" * 60)
summary_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k != 'model'}
    for name, vals in results.items()
}).T
print(summary_df.to_string())

# ─── 8. Pick Best Model ───────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['accuracy'])
best_result = results[best_name]
best_model  = best_result['model']

print(f"\n Best model: {best_name}  (accuracy={best_result['accuracy']:.4f})")

# ─── 9. Hyperparameter Tuning (best model only) ───────────────────────────────
print("\n" + "─" * 60)
print(f"HYPERPARAMETER TUNING — {best_name}")
print("─" * 60)

if best_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }
elif best_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
    }
else:  # SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
    }

grid_search = GridSearchCV(
    best_model.__class__(**({'probability': True, 'random_state': 42}
                             if best_name == 'SVM (RBF)' else {'random_state': 42})),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train_sc, y_train)

tuned_model = grid_search.best_estimator_
tuned_acc = accuracy_score(y_test, tuned_model.predict(X_test_sc))

print(f"Best params: {grid_search.best_params_}")
print(f"Tuned accuracy: {tuned_acc:.4f}")

final_model = tuned_model if tuned_acc >= best_result['accuracy'] else best_model
print(f"\nFinal model accuracy: {accuracy_score(y_test, final_model.predict(X_test_sc)):.4f}")

# ─── 10. Save Model & Scaler ─────────────────────────────────────────────────
model_path   = os.path.join(ML_DIR, 'best_model.pkl')
scaler_path  = os.path.join(ML_DIR, 'scaler.pkl')
feature_path = os.path.join(ML_DIR, 'feature_names.pkl')

with open(model_path,   'wb') as f: pickle.dump(final_model, f)
with open(scaler_path,  'wb') as f: pickle.dump(scaler, f)
with open(feature_path, 'wb') as f: pickle.dump(FEATURES, f)

print(f"\n Model saved  → {model_path}")
print(f" Scaler saved → {scaler_path}")
print(f" Features saved → {feature_path}")

# ─── 11. Feature Importance (if tree-based) ──────────────────────────────────
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    print(f"\nFeature Importances ({best_name}):")
    for i in idx:
        print(f"  {FEATURES[i]:35s}: {importances[i]:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
