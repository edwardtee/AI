from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, cross_val_score, KFold, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, make_scorer,  roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import joblib
import seaborn as sns
from sklearn.impute import SimpleImputer

df = pd.read_csv("C:/Users/edwar/Downloads/archive/heart_disease_dataset.csv")
# ==========================
# Separate Features and Target
# ==========================

#X = df.drop(columns=["Heart Disease"])
X = df[["Age", "Cholesterol"]]
y = df["Heart Disease"]
from pathlib import Path


# ==========================
# Detect Numerical and Categorical Columns
# ==========================

numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["str"]).columns

print("Numerical Columns:")
print(numerical_columns)
print("\nCategorical Columns:")
print(categorical_columns)

# ==========================
# Handle Missing Values
# ==========================

num_imputer = SimpleImputer(strategy="median")
X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])

# ==========================
# Outlier Removal
# ==========================

mask = pd.Series(True, index=X.index)
for col in numerical_columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = X[(X[col] < lower) | (X[col] > upper)]
    print(col, len(outliers))
    mask &= (X[col] >= lower) & (X[col] <= upper)

X = X[mask]
y = y[mask]

# ==========================
# One-Hot Encoding
# ==========================

X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics collected for every GridSearchCV below. refit="accuracy" is what
# GridSearchCV uses to pick best_params_ / best_index_; the other four are
# just along for the ride so we can report them at whichever index we need.
MULTI_SCORING = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0)
}

MODEL_DIR = Path("Tuning Result") / "Old Dataset LR'C TF"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def plot_param_curve(param_name, param_values, mean_scores, std_scores,
                      best_value, xlabel, filename, logx=False):
    """Plot mean CV accuracy vs a single hyperparameter."""
    plt.figure(figsize=(8, 6))
    if logx:
        plt.semilogx(param_values, mean_scores, marker="o")
    else:
        plt.plot(param_values, mean_scores, marker="o")
    plt.fill_between(
        param_values,
        np.array(mean_scores) - np.array(std_scores),
        np.array(mean_scores) + np.array(std_scores),
        alpha=0.2
    )
    plt.axvline(best_value, color="red", linestyle="--",
                label=f"Best {param_name} = {best_value}")
    plt.title(f"GridSearchCV Accuracy vs {param_name}")
    plt.xlabel(xlabel)
    plt.ylabel("Mean CV Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def plot_categorical_curve(param_name, categories, mean_scores, std_scores,
                            best_value, filename):
    """Bar plot of mean CV accuracy for a categorical hyperparameter."""
    plt.figure(figsize=(7, 5))
    colors = ["red" if c == best_value else "steelblue" for c in categories]
    plt.bar(categories, mean_scores, yerr=std_scores, capsize=5, color=colors)
    plt.title(f"GridSearchCV Accuracy vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Mean CV Accuracy")
    plt.grid(axis="y")
    plt.savefig(MODEL_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


def print_stage_metrics(cv_results, idx, label):
    """Print Accuracy, ROC-AUC, Recall, F1-score, and Precision for one
    specific row (parameter combination) of a GridSearchCV's cv_results_."""
    print(f"-- {label}: CV metrics at this setting --")
    print(f"Accuracy : {cv_results['mean_test_accuracy'][idx]*100:.2f}%")
    print(f"ROC-AUC  : {cv_results['mean_test_roc_auc'][idx]:.4f}")
    print(f"Recall   : {cv_results['mean_test_recall'][idx]:.4f}")
    print(f"F1-score : {cv_results['mean_test_f1'][idx]:.4f}")
    print(f"Precision: {cv_results['mean_test_precision'][idx]:.4f}\n")

def save_cv_results(grid, filename, param_cols=None):
    """Save full cv_results_ (all tested parameter values, train/test
    mean & std for every scorer, rank) to CSV for later inspection."""
    results_df = pd.DataFrame(grid.cv_results_)

    # Put the most useful columns first if specified, keep everything else too
    if param_cols:
        front = [c for c in param_cols if c in results_df.columns]
        rest = [c for c in results_df.columns if c not in front]
        results_df = results_df[front + rest]

    results_df.to_csv(MODEL_DIR / filename, index=False)
    print(f"Saved full grid results: {MODEL_DIR / filename}")
    return results_df


# ==========================================================
# STEP 1: Tune KNN n_neighbors (K = 3 to 30)
# Base settings held fixed: weights='uniform', metric='minkowski'
# ==========================================================

print("\n=== STEP 1: Tuning KNN n_neighbors (3-30) ===")

k_range = list(range(3, 31))
knn_k_grid = GridSearchCV(
    estimator=KNeighborsClassifier(weights="uniform", metric="minkowski"),
    param_grid={"n_neighbors": k_range},
    cv=cv,
    scoring=MULTI_SCORING,
    refit="accuracy",
    n_jobs=-1
)
knn_k_grid.fit(X_train, y_train)
save_cv_results(
    knn_k_grid, "cv_results_knn_k.csv",
    param_cols=["param_n_neighbors", "mean_train_accuracy", "mean_test_accuracy",
                "std_test_accuracy", "mean_test_roc_auc", "mean_test_recall",
                "mean_test_f1", "mean_test_precision", "rank_test_accuracy"]
)

best_k = knn_k_grid.best_params_["n_neighbors"]
k_mean_scores = knn_k_grid.cv_results_["mean_test_accuracy"]
k_std_scores = knn_k_grid.cv_results_["std_test_accuracy"]

print(f"Best n_neighbors: {best_k} (CV Accuracy = {knn_k_grid.best_score_:.4f})")
print_stage_metrics(knn_k_grid.cv_results_, knn_k_grid.best_index_, "KNN n_neighbors (best K)")

plot_param_curve(
    "n_neighbors", k_range, k_mean_scores, k_std_scores, best_k,
    xlabel="Number of Neighbors (K)",
    filename="tuning_knn_k.png"
)

# ==========================================================
# STEP 2: Tune KNN weights (uniform vs distance), using best_k
# NOTE: per requirement, we force weights='uniform' in the final
# model regardless of which weighting scores higher in the grid
# search, since 'distance' tends to overfit (each point fits its
# own nearest neighbors too closely, especially for lower K).
# ==========================================================

print("\n=== STEP 2: Tuning KNN weights (uniform vs distance) ===")

knn_weights_grid = GridSearchCV(
    estimator=KNeighborsClassifier(n_neighbors=best_k, metric="minkowski"),
    param_grid={"weights": ["uniform", "distance"]},
    cv=cv,
    scoring=MULTI_SCORING,
    refit="accuracy",
    return_train_score=True,
    n_jobs=-1
)
knn_weights_grid.fit(X_train, y_train)

save_cv_results(
    knn_weights_grid, "cv_results_knn_weights.csv",
    param_cols=["param_weights", "mean_train_accuracy", "mean_test_accuracy",
                "std_test_accuracy", "mean_test_roc_auc", "mean_test_recall",
                "mean_test_f1", "mean_test_precision", "rank_test_accuracy"]
)

weights_categories = knn_weights_grid.cv_results_["param_weights"].data.tolist()
weights_mean_scores = knn_weights_grid.cv_results_["mean_test_accuracy"]
weights_std_scores = knn_weights_grid.cv_results_["std_test_accuracy"]
grid_best_weights = knn_weights_grid.best_params_["weights"]
weights_train_scores = knn_weights_grid.cv_results_["mean_train_accuracy"]

print(f"GridSearchCV top-scoring weights: {grid_best_weights} "
      f"(CV Accuracy = {knn_weights_grid.best_score_:.4f})")

# --- Train vs Test (CV) accuracy gap for each weighting scheme ---
print("\n-- Train vs CV-Test Accuracy Gap by weights --")
for cat, train_acc, test_acc in zip(weights_categories, weights_train_scores, weights_mean_scores):
    gap = train_acc - test_acc
    print(f"{cat:10s} | Train Acc: {train_acc*100:.2f}%  "
          f"| CV Test Acc: {test_acc*100:.2f}%  | Gap: {gap*100:.2f} pts")

# Force uniform to avoid overfitting, even if 'distance' scored higher
best_weights = "uniform"
uniform_idx = weights_categories.index("uniform")
uniform_score = weights_mean_scores[uniform_idx]
print(f"Using weights='uniform' by design (CV Accuracy = {uniform_score:.4f})")
print_stage_metrics(knn_weights_grid.cv_results_, uniform_idx, "KNN weights (uniform, forced)")

plot_categorical_curve(
    "weights", weights_categories, weights_mean_scores, weights_std_scores,
    best_value=grid_best_weights,  # highlight the grid's top scorer in red
    filename="tuning_knn_weights.png"
)

# ==========================================================
# STEP 3: Tune KNN metric (minkowski, euclidean, manhattan)
# using best_k and best_weights ('uniform')
# ==========================================================

print("\n=== STEP 3: Tuning KNN metric ===")

knn_metric_grid = GridSearchCV(
    estimator=KNeighborsClassifier(n_neighbors=best_k, weights=best_weights),
    param_grid={"metric": ["minkowski", "euclidean", "manhattan"]},
    cv=cv,
    scoring=MULTI_SCORING,
    refit="accuracy",
    n_jobs=-1
)
knn_metric_grid.fit(X_train, y_train)
save_cv_results(
    knn_metric_grid, "cv_results_knn_metric.csv",
    param_cols=["param_metric", "mean_train_accuracy", "mean_test_accuracy",
                "std_test_accuracy", "mean_test_roc_auc", "mean_test_recall",
                "mean_test_f1", "mean_test_precision", "rank_test_accuracy"]
)

metric_categories = knn_metric_grid.cv_results_["param_metric"].data.tolist()
metric_mean_scores = knn_metric_grid.cv_results_["mean_test_accuracy"]
metric_std_scores = knn_metric_grid.cv_results_["std_test_accuracy"]
best_metric = knn_metric_grid.best_params_["metric"]

print(f"Best metric: {best_metric} (CV Accuracy = {knn_metric_grid.best_score_:.4f})")
print_stage_metrics(knn_metric_grid.cv_results_, knn_metric_grid.best_index_, "KNN metric (best)")

plot_categorical_curve(
    "metric", metric_categories, metric_mean_scores, metric_std_scores,
    best_value=best_metric,
    filename="tuning_knn_metric.png"
)

# ==========================================================
# STEP 4: Tune Logistic Regression C (0.001 to 10)
# ==========================================================

print("\n=== STEP 4: Tuning Logistic Regression C (0.001-10) ===")

c_range = [0.001, 0.01, 0.1, 1, 10, 100]
lr_c_grid = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid={"C": c_range},
    cv=cv,
    scoring=MULTI_SCORING,
    refit="accuracy",
    n_jobs=-1
)
lr_c_grid.fit(X_train, y_train)
save_cv_results(
    lr_c_grid, "cv_results_lr_c.csv",
    param_cols=["param_C", "mean_train_accuracy", "mean_test_accuracy",
                "std_test_accuracy", "mean_test_roc_auc", "mean_test_recall",
                "mean_test_f1", "mean_test_precision", "rank_test_accuracy"]
)

best_C = lr_c_grid.best_params_["C"]
c_mean_scores = lr_c_grid.cv_results_["mean_test_accuracy"]
c_std_scores = lr_c_grid.cv_results_["std_test_accuracy"]

print(f"Best C: {best_C:.5f} (CV Accuracy = {lr_c_grid.best_score_:.4f})")
print_stage_metrics(lr_c_grid.cv_results_, lr_c_grid.best_index_, "Logistic Regression C (best)")

plot_param_curve(
    "C", c_range, c_mean_scores, c_std_scores, best_C,
    xlabel="C (Inverse Regularization Strength)",
    filename="tuning_lr_c.png",
    logx=True
)

# ==========================================================
# FINAL MODEL: Build the stacking classifier with the tuned
# parameters found above, then evaluate.
# ==========================================================

print("\n=== Building Final Model with Optimal Parameters ===")
print(f"KNN  -> n_neighbors={best_k}, weights='{best_weights}', metric='{best_metric}'")
print(f"LR   -> C={best_C:.5f}")

base_models = [
    ("lr", LogisticRegression(random_state=42, C=best_C)),
    ("knn", KNeighborsClassifier(
        n_neighbors=best_k,
        weights=best_weights,
        metric=best_metric
    ))
]

heart_net = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(C=1),
    stack_method="predict_proba",
    cv=5
)

cv_result = cross_validate(
    estimator=heart_net,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring=["accuracy", "roc_auc", "recall", "f1", "precision"],
    return_train_score=False
)

heart_net.fit(X_train, y_train)

y_pred = heart_net.predict(X_test)
y_prob = heart_net.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===============================")
print(" Optimally-Tuned Hybrid KNN + LR Results")
print("===============================\n")
print("-- Test Set Metrics --")
print(f"Accuracy : {accuracy*100:.2f}%")
print(f"ROC-AUC  : {auc:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Precision: {precision:.4f}\n")
print("-- Mean Cross-Validation Metrics (Training Set) --")
print(f"Mean CV Accuracy : {cv_result['test_accuracy'].mean()*100:.2f}%")
print(f"Mean CV ROC-AUC  : {cv_result['test_roc_auc'].mean():.4f}")
print(f"Mean CV Recall   : {cv_result['test_recall'].mean():.4f}")
print(f"Mean CV F1-Score : {cv_result['test_f1'].mean():.4f}")
print(f"Mean CV Precision: {cv_result['test_precision'].mean():.4f}")

# ==========================================================
# Summary table: Accuracy, ROC-AUC, Recall, F1-score, and
# Precision achieved at each tuning stage plus the final model
# ==========================================================

summary_rows = [
    ("KNN n_neighbors", knn_k_grid.cv_results_, knn_k_grid.best_index_),
    ("KNN weights (uniform, forced)", knn_weights_grid.cv_results_, uniform_idx),
    ("KNN metric", knn_metric_grid.cv_results_, knn_metric_grid.best_index_),
    ("LR C", lr_c_grid.cv_results_, lr_c_grid.best_index_),
]

summary_table = pd.DataFrame([
    {
        "Stage": name,
        "Accuracy": cr["mean_test_accuracy"][idx],
        "ROC-AUC": cr["mean_test_roc_auc"][idx],
        "Recall": cr["mean_test_recall"][idx],
        "F1-score": cr["mean_test_f1"][idx],
        "Precision": cr["mean_test_precision"][idx],
    }
    for name, cr, idx in summary_rows
])

summary_table = pd.concat([summary_table, pd.DataFrame([{
    "Stage": "Final Model (CV)",
    "Accuracy": cv_result["test_accuracy"].mean(),
    "ROC-AUC": cv_result["test_roc_auc"].mean(),
    "Recall": cv_result["test_recall"].mean(),
    "F1-score": cv_result["test_f1"].mean(),
    "Precision": cv_result["test_precision"].mean(),
}])], ignore_index=True)

print("\n=== Summary: Metrics at Each Tuning Stage ===")
print(summary_table.to_string(index=False))
summary_table.to_csv(MODEL_DIR / "tuning_stage_metrics.csv", index=False)
print(f"Saved table: {MODEL_DIR / 'tuning_stage_metrics.csv'}")

# Accuracy line chart across tuning stages (kept from before)
stage_labels = ["KNN K", "KNN weights\n(uniform, forced)", "KNN metric", "LR C", "Final Model"]
stage_scores = list(summary_table["Accuracy"])

plt.figure(figsize=(9, 6))
plt.plot(stage_labels, stage_scores, marker="o", linewidth=2)
for i, s in enumerate(stage_scores):
    plt.text(i, s, f"{s:.4f}", ha="center", va="bottom")
plt.title("CV Accuracy Across Sequential Tuning Stages")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)
plt.savefig(MODEL_DIR / "tuning_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved plot: tuning_summary.png")

# ==========================================================
# Learning Curve: how the final tuned model's accuracy scales
# with training set size (train vs cross-validation score)
# ==========================================================

print("\n=== Generating Learning Curve for Final Tuned Model ===")

train_sizes, train_scores, valid_scores = learning_curve(
    heart_net,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 9),
    random_state=42,
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
valid_mean = valid_scores.mean(axis=1)
valid_std = valid_scores.std(axis=1)

print("Train sizes     :", train_sizes)
print("Train accuracy  :", np.round(train_mean, 4))
print("Valid accuracy  :", np.round(valid_mean, 4))

plt.figure(figsize=(8, 6))

plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)

plt.plot(train_sizes, valid_mean, marker="s", label="Validation Accuracy")
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.15)

plt.title("Learning Curve - Optimally-Tuned Hybrid KNN + LR")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(MODEL_DIR / "tuned_learning_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved plot: tuned_learning_curve.png")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Heart Disease"])

# Plot with a blue color map
disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix", fontsize=14, fontweight='bold')

# Save the plot (Optional)
plt.savefig(MODEL_DIR / "NewDataset_confusion_matrix_sklearn.png", dpi=300, bbox_inches='tight')

fpr, tpr, thresholds = roc_curve(y_test, y_prob)


plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(MODEL_DIR / "NewDataset_AUC-ROC.png", dpi=300)