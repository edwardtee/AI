# ==========================================================
# Heart Disease Prediction using Hybrid RF + SVM + LR
# Based on:
# A Hybrid Random Forest–Support Vector Machine Model
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, learning_curve

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import joblib

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("C:/Users/edwar/Downloads/archive/heart_disease_dataset.csv")

# ==========================
# Separate Features and Target
# ==========================

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# ==========================
# Detect Numerical and Categorical Columns
# ==========================

numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns

categorical_columns = X.select_dtypes(include=["object"]).columns

print("Numerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Handle Missing Values
# ==========================

# Numerical Features

num_imputer = SimpleImputer(strategy="median")

X_train[numerical_columns] = num_imputer.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = num_imputer.transform(
    X_test[numerical_columns]
)

# Categorical Features

cat_imputer = SimpleImputer(strategy="most_frequent")

X_train[categorical_columns] = cat_imputer.fit_transform(
    X_train[categorical_columns]
)

X_test[categorical_columns] = cat_imputer.transform(
    X_test[categorical_columns]
)

# ==========================
# One-Hot Encoding
# ==========================

X_train = pd.get_dummies(
    X_train,
    columns=categorical_columns,
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_columns,
    drop_first=True
)

# Ensure both datasets have identical columns

X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

# ==========================
# Train Random Forest
# ==========================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_train_prob = rf.predict_proba(X_train)

rf_test_prob = rf.predict_proba(X_test)

# ==========================
# Train SVM
# ==========================

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm.fit(X_train, y_train)

svm_train_prob = svm.predict_proba(X_train)

svm_test_prob = svm.predict_proba(X_test)

# ==========================
# Create Meta Features
# ==========================

meta_train = np.hstack((
    rf_train_prob,
    svm_train_prob
))

meta_test = np.hstack((
    rf_test_prob,
    svm_test_prob
))

# ==========================
# Train Logistic Regression
# ==========================

meta_model = LogisticRegression(
    random_state=42
)

meta_model.fit(
    meta_train,
    y_train
)

# ==========================
# Final Prediction
# ==========================

y_pred = meta_model.predict(meta_test)

y_prob = meta_model.predict_proba(meta_test)[:,1]

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

print("\n===============================")
print(" Hybrid RF + SVM + LR Results")
print("===============================\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

joblib.dump(meta_model, "rf_svm_lr.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(num_imputer, "num_imputer.pkl")
joblib.dump(cat_imputer, "cat_imputer.pkl")
joblib.dump(scaler, "rf_scaler.pkl")
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")
print("Model and Encoders saved successfully!")