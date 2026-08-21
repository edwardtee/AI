import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# Paths
# ============================================================
# Adjust MODEL_DIR if your saved_models folder lives elsewhere
# relative to wherever you run `streamlit run app.py` from.
MODEL_DIR = Path("saved_models")
SCALER_PATH = MODEL_DIR / "heart_scaler.joblib"
COLUMNS_PATH = MODEL_DIR / "heart_feature_columns.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Prediction System")
st.write(
    """
    Pick a model, describe the patient's health profile, and get a live
    risk prediction. All fields update the prediction instantly — no
    button needed.
    """
)

# ============================================================
# Discover available models
# ============================================================
# Any *_model.joblib file in saved_models is treated as a selectable model.
if not MODEL_DIR.exists():
    st.error(f"Model directory not found: {MODEL_DIR.resolve()}")
    st.stop()

model_files = sorted(MODEL_DIR.glob("*_model.joblib"))

if not model_files:
    st.error(f"No '*_model.joblib' files found in {MODEL_DIR.resolve()}")
    st.stop()

# Friendly names for known model file prefixes.
# knn_lr_model.joblib -> "KNN + Logistic Regression (Stacked)"
# svm_rf_lr_model.joblib -> "SVM + Random Forest + LR (Stacked)"
FRIENDLY_NAMES = {
    "knn_lr": "KNN + Logistic Regression (Stacked)",
    "svm_rf_lr": "SVM + Random Forest + LR (Stacked)",
}

MODEL_OPTIONS = {
    FRIENDLY_NAMES.get(p.stem.replace("_model", "").lower(), p.stem.replace("_model", "").upper()): p
    for p in model_files
}

# ============================================================
# Cached loaders (avoid re-reading disk on every slider move)
# ============================================================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource
def load_scaler(path: str):
    return joblib.load(path)


@st.cache_resource
def load_columns(path: str):
    return joblib.load(path)


@st.cache_data
def load_metrics(path: str):
    """Load precomputed classification metrics per model, if available.

    Expected format:
    {
        "KNN + Logistic Regression (Stacked)": {
            "accuracy": 0.87, "precision": 0.84, "recall": 0.81,
            "f1": 0.82, "roc_auc": 0.91
        },
        "SVM + Random Forest + LR (Stacked)": {...},
    }
    """
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


try:
    scaler = load_scaler(str(SCALER_PATH))
    feature_columns = load_columns(str(COLUMNS_PATH))
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.caption(
        "Run your training script(s) first — they need to `joblib.dump()` "
        "the scaler and the feature-column list, shared across both models."
    )
    st.stop()

metrics = load_metrics(str(METRICS_PATH))

# ============================================================
# Model selection
# ============================================================
model_choice = st.selectbox("Select a model", list(MODEL_OPTIONS.keys()))

try:
    model = load_model(str(MODEL_OPTIONS[model_choice]))
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_OPTIONS[model_choice].resolve()}")
    st.stop()

# ============================================================
# Metrics expander
# ============================================================
with st.expander("📊 Model Performance"):
    if metrics is None:
        st.info(
            f"No metrics file found at `{METRICS_PATH}`. "
            "Run your updated training script(s) to generate one, "
            "then re-run this app."
        )
    elif model_choice not in metrics:
        st.warning(f"No stored metrics for '{model_choice}' in metrics.json.")
    else:
        m = metrics[model_choice]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        col2.metric("F1-score", f"{m['f1']:.4f}")
        col3.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        col4, col5 = st.columns(2)
        col4.metric("Precision", f"{m['precision']:.4f}")
        col5.metric("Recall", f"{m['recall']:.4f}")

# ============================================================
# Training ranges (numerical features) — used for slider bounds
# Pulled from your describe() output. Update these if your split differs.
# ============================================================
AGE_MIN, AGE_MAX = 25, 79
CHOL_MIN, CHOL_MAX = 150, 349
BP_MIN, BP_MAX = 90, 179
HR_MIN, HR_MAX = 60, 99
EXERCISE_MIN, EXERCISE_MAX = 0, 9
STRESS_MIN, STRESS_MAX = 1, 10
SUGAR_MIN, SUGAR_MAX = 70, 199

# ============================================================
# User inputs — numerical (left 2 columns) + categorical (right 2 columns)
# ============================================================
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", AGE_MIN, AGE_MAX, 52, 1)
    cholesterol = st.slider("Cholesterol", CHOL_MIN, CHOL_MAX, 250, 1)
    blood_pressure = st.slider("Blood Pressure", BP_MIN, BP_MAX, 135, 1)
    heart_rate = st.slider("Heart Rate", HR_MIN, HR_MAX, 79, 1)

with col2:
    exercise_hours = st.slider("Exercise Hours (per week)", EXERCISE_MIN, EXERCISE_MAX, 5, 1)
    stress_level = st.slider("Stress Level (1-10)", STRESS_MIN, STRESS_MAX, 6, 1)
    blood_sugar = st.slider("Blood Sugar", SUGAR_MIN, SUGAR_MAX, 135, 1)

col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking Status", ["Never", "Current", "Former"])
    alcohol = st.selectbox("Alcohol Intake", ["Never", "Moderate", "Heavy"])
    family_history = st.selectbox("Family History of Heart Disease?", ["No", "Yes"])

with col4:
    diabetes = st.selectbox("Diabetes Diagnosis?", ["No", "Yes"])
    obesity = st.selectbox("Obesity?", ["No", "Yes"])
    angina = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Non-anginal Pain", "Typical Angina", "Atypical Angina", "Asymptomatic"]
    )

# ============================================================
# Live prediction
# ============================================================
# Column names here MUST match the raw column names in your CSV exactly
# (case + spacing), since they're what gets passed into pd.get_dummies().
# Double-check these against your dataset headers — the names below are
# my best guess based on your describe() output and selectbox labels.
raw_input = pd.DataFrame({
    "Age": [age],
    "Cholesterol": [cholesterol],
    "Blood Pressure": [blood_pressure],
    "Heart Rate": [heart_rate],
    "Exercise Hours": [exercise_hours],
    "Stress Level": [stress_level],
    "Blood Sugar": [blood_sugar],
    "Gender": [gender],
    "Smoking": [smoking],
    "Alcohol Intake": [alcohol],
    "Family History": [family_history],
    "Diabetes": [diabetes],
    "Obesity": [obesity],
    "Exercise Induced Angina": [angina],
    "Chest Pain Type": [chest_pain],
})

numerical_columns = [
    "Age", "Cholesterol", "Blood Pressure", "Heart Rate",
    "Exercise Hours", "Stress Level", "Blood Sugar",
]
categorical_columns = [
    "Gender", "Smoking", "Alcohol Intake", "Family History",
    "Diabetes", "Obesity", "Exercise Induced Angina", "Chest Pain Type",
]

try:
    # Single-row input encoding: do NOT use drop_first=True here because pandas will drop
    # the single category present in a 1-row DataFrame, making all dummy features 0.
    encoded = pd.get_dummies(raw_input, columns=categorical_columns)

    # Align to the exact column set/order the model was trained on.
    # Any dummy column not produced by this single row (e.g. a category
    # that isn't the current selection) gets filled with 0, matching
    # what get_dummies would have produced during training.
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale only the numerical columns, same as training.
    encoded[numerical_columns] = scaler.transform(encoded[numerical_columns])

    prediction = model.predict(encoded)[0]
    label = "High Risk" if prediction == 1 else "Low Risk"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(encoded)[0][1]
        if label == "High Risk" and proba > 0.5:
            st.warning(
                f"Prediction ({model_choice}): **{label}** "
                f"— estimated probability of heart disease: {proba:.1%}"
            )
        else:
            st.success(
                f"Prediction ({model_choice}): **{label}** "
                f"— estimated probability of heart disease: {proba:.1%}"
            )
    else:
        st.success(f"Prediction ({model_choice}): **{label}**")

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.caption(
        "This usually means the feature columns/order, or the category "
        "label spelling, doesn't match what your training script used. "
        "Check `feature_columns` (heart_feature_columns.joblib) against "
        "what pd.get_dummies() produced for this row."
    )

# py -m streamlit run app.py