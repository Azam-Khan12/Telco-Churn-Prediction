import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Model selection dropdown
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression (Recommended)", "Naive Bayes (High Recall)"])

# Load selected model
if model_choice == "Logistic Regression (Recommended)":
    model_file = 'models/churn_model_logistic.pkl'
    scaler_file = 'models/scaler_logistic.pkl'
    info_file = 'models/preprocess_info_logistic.pkl'
    use_scaler = True
else:
    model_file = 'models/churn_model_naive.pkl'
    info_file = 'models/preprocess_info_naive.pkl'
    use_scaler = False

with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(info_file, 'rb') as f:
    info = pickle.load(f)

if use_scaler:
    with open('models/scaler_logistic.pkl', 'rb') as f:
        scaler = pickle.load(f)

categorical_cols = info['categorical_cols']
feature_names = info['feature_names']

st.title("🏢 Telco Customer Churn Prediction")
st.markdown(f"**Current Model: {model_choice}**")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.05)

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    total_charges = tenure * monthly
    st.info(f"🔢 Auto-Calculated Total Charges: ${total_charges:.2f}")

    data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([data])
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_names]
    
    if use_scaler:
        input_final = scaler.transform(input_encoded)
    else:
        input_final = input_encoded
    
    probability = model.predict_proba(input_final)[0][1]
    prediction = model.predict(input_final)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 High Risk of Churn")
        st.warning(f"Churn Probability: {probability:.2%}")
        st.info("💡 Suggestion: Call the customer, offer discount or long-term contract!")
    else:
        st.success("✅ Low Risk – Likely to Stay")
        st.success(f"Churn Probability: {probability:.2%}")
        st.info("👍 Keep up the good service!")