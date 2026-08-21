import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report,  roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import joblib
import seaborn as sns
from scipy.stats import zscore
import math

df = pd.read_csv(r"C:\Users\edwar\Downloads\cleaned_merged_heart_dataset.csv")
print(df.isnull().sum())       # Alcohol Intake got 340 row is None, which python will assume it as missing value.
print(df.info())
'''print("Checking for duplicate rows in the dataset...")
duplicate_groups = (
    df.groupby(df.columns.tolist())
      .size()
      .reset_index(name='Count')
)

print(duplicate_groups.sort_values('Count', ascending=False).head(50))
#df.drop_duplicates(inplace=True)  # Remove duplicate rows if any
print(df.describe())
for i in df.columns:

  print(i)

  print(df[i].unique())'''

'''# ==========================
# Target Distribution Plot
# ==========================

# Target distribution
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable (Heart Disease)')
plt.xlabel('Heart Disease (0 = Less Chance, 1 = More Chance)')
plt.ylabel('Count')
plt.savefig("target_distribution_bar.png", dpi=300)
plt.show()
print("Plot saved → Distribution_of_Target.png")'''
X=df.drop(columns=["target"])
y=df["target"]
'''mask = pd.Series(True, index=X.index)
for col in ["age", "chol", "oldpeak", "thalachh", "trestbps"]:

    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = X[(X[col] < lower) |
                        (X[col] > upper)]

    print(col, len(outliers))

    mask &= (X[col] >= lower) & (X[col] <= upper)
X = X[mask]
y = y[mask]'''
'''num_cols = ["age","sex","cp", "trestbps","chol", "fbs","restecg", "thalachh", "exang", "oldpeak", "slope", "ca"]
fig, axes = plt.subplots(3, 4, figsize=(18, 8))

axes = axes.flatten()

for i, column in enumerate(num_cols):

    sns.boxplot(
        y=df[column],
        ax=axes[i]
    )

    axes[i].set_title(column, fontsize=12)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

plt.suptitle("Boxplots of Numerical Features", fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("Numerical_Boxplots.png", dpi=300)'''


'''# Numerical features
num_cols = ["age", "chol", "oldpeak", "thalachh", "trestbps", "ca", "thal"]

# Calculate Z-scores
z_scores = df[num_cols].apply(zscore)

# Number of rows and columns
n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(z_scores[col].dropna(), bins=30, edgecolor='black')
    axes[i].axvline(3, color='red', linestyle='--', label='Z=3')
    axes[i].axvline(-3, color='red', linestyle='--', label='Z=-3')
    axes[i].set_title(f"{col} Z-score Distribution")
    axes[i].set_xlabel("Z-score")
    axes[i].set_ylabel("Frequency")

# Remove unused subplots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("zscore_all_numerical_features.png", dpi=300)
plt.show()'''

print(df.describe())
df['ca'] = df['ca'].replace(4, 0)
print(df.describe())