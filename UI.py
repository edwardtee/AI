import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Used to load your model and encoders

st.title("❤️ Heart Disease Risk Prediction Form")
st.write("Please fill in the patient details below to perform a prediction.")

model_choice = st.selectbox(
    "Select Prediction Model",
    (
        "Hybrid RF + SVM + LR",
        "Hybrid KNN + LR"
    )
)

# 1. Load the model and encoders securely using Streamlit caching
@st.cache_resource
def load_models():

    # KNN + LR
    knn_model = joblib.load("heart_net.pkl")
    knn_encoders = joblib.load("label_encoders.pkl")
    knn_scaler = joblib.load("scaler.pkl")

    # RF + SVM + LR
    rf = joblib.load("rf_model.pkl")
    svm = joblib.load("svm_model.pkl")
    rf_svm_lr = joblib.load("rf_svm_lr.pkl")

    num_imputer = joblib.load("num_imputer.pkl")
    cat_imputer = joblib.load("cat_imputer.pkl")
    scaler = joblib.load("rf_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return (
        knn_model,
        knn_encoders,
        knn_scaler,
        rf,
        svm,
        rf_svm_lr,
        num_imputer,
        cat_imputer,
        scaler,
        feature_columns
    )
(
    heart_net,
    saved_encoders,
    knn_scaler,
    rf,
    svm,
    rf_svm_lr,
    num_imputer,
    cat_imputer,
    rf_scaler,
    feature_columns
) = load_models()

# --- Your existing Column Layout UI Code stays exactly the same ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Numerical Data")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=500, value=200)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=220, value=75)
    exercise_hours = st.number_input("Weekly Exercise Hours", min_value=0, max_value=168, value=3)
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    blood_sugar = st.number_input("Blood Sugar Level (mg/dL)", min_value=50, max_value=400, value=90)

with col2:
    st.subheader("Categorical Data")
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking Status", ["Never", "Current", "Former"])
    alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])
    family_history = st.selectbox("Family History of Heart Disease?", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes Diagnosis?", ["No", "Yes"])
    obesity = st.selectbox("Obesity?", ["No", "Yes"])
    angina = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain Type", ["Non-anginal Pain", "Typical Angina", "Atypical Angina", "Asymptomatic"])


# 2. Trigger prediction when user clicks the button
if st.button("Generate Prediction Data"):
    
    # Capture all inputs into a raw dataframe
    user_input_df = pd.DataFrame([{
        "Age": age, "Gender": gender, "Cholesterol": cholesterol,
        "Blood Pressure": blood_pressure, "Heart Rate": heart_rate,
        "Smoking": smoking, "Alcohol Intake": alcohol,
        "Exercise Hours": exercise_hours, "Family History": family_history,
        "Diabetes": diabetes, "Obesity": obesity,
        "Stress Level": stress_level, "Blood Sugar": blood_sugar,
        "Exercise Induced Angina": angina, "Chest Pain Type": chest_pain
    }])

    if model_choice == "Hybrid KNN + LR":
        # 3. Filter down to the exact features the hybrid model expects
        important_features = ["Age", "Cholesterol", "Gender"]
        X_new = user_input_df[important_features].copy()
        
        # 4. Encode categorical features using your SAVED encoders (No fit_transform!)
        for col in X_new.select_dtypes(include=['str', 'string', 'object']).columns:
            if col in saved_encoders:
                le = saved_encoders[col]
                X_new[col] = le.transform(X_new[col])  # Crucial: .transform(), NOT .fit_transform()

        # 4.5 SCALE THE DATA (Crucial Step!)
        # StackingClassifier internally calls each base model, all of which need scaled input
        num_cols = ["Age", "Cholesterol"]

        X_new[num_cols] = knn_scaler.transform(
            X_new[num_cols]
        )

        # 5. Make the Prediction using the hybrid Stacking model
        prediction = heart_net.predict(X_new)
        prediction_proba = heart_net.predict_proba(X_new)
    else:
        X_new = user_input_df.copy()
        numerical_columns = X_new.select_dtypes(include=["int64", "float64"]).columns
        categorical_columns = X_new.select_dtypes(include=["str"]).columns
        

        # Missing values
        X_new[numerical_columns] = num_imputer.transform(
            X_new[numerical_columns]
        )

        X_new[categorical_columns] = cat_imputer.transform(
            X_new[categorical_columns]
        )

        # One-hot encoding
        X_new = pd.get_dummies(
            X_new,
            columns=categorical_columns,
            drop_first=True
        )

        # Match training columns
        X_new = X_new.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Scale numerical columns
        X_new[numerical_columns] = rf_scaler.transform(
            X_new[numerical_columns]
        )
        rf_prob = rf.predict_proba(X_new)
        svm_prob = svm.predict_proba(X_new)
        meta_features = np.hstack((rf_prob, svm_prob))
        prediction = rf_svm_lr.predict(meta_features)

        prediction_proba = rf_svm_lr.predict_proba(meta_features)
    
    
    
    
    # 6. Display the Results beautifully
    st.write("---")
    st.subheader("📊 Prediction Results")
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Confidence: {prediction_proba[0][1]*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Confidence: {prediction_proba[0][0]*100:.1f}%)")

#py -m streamlit run UI.py