import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page config - Imperial & Energetic Theme
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Imperial, Classy & Energetic Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(to bottom, #1a1a2e, #16213e);
        color: #e94560;
    }
    .sidebar .sidebar-content {
        background: #0f3460;
    }
    h1 {
        color: #e94560;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        text-shadow: 0 0 10px #e94560;
    }
    .stSelectbox > div > div > div {
        color: white;
        background-color: #16213e;
    }
    .stButton > button {
        background-color: #e94560;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
        box-shadow: 0 0 15px #ff6b6b;
    }
    .stSuccess {
        background-color: #28a745 !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 1rem;
        border-radius: 10px;
    }
    .stError {
        background-color: #dc3545 !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 1rem;
        border-radius: 10px;
    }
    .stWarning {
        background-color: #ffc107;
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Model Selection
st.sidebar.title("🔥 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Logistic Regression (Recommended)", "Naive Bayes (High Recall)"]
)

# Correct paths for Streamlit Cloud
if model_choice == "Logistic Regression (Recommended)":
    model_file = "models/churn_model_logistic.pkl"
    scaler_file = "models/scaler_logistic.pkl"
    info_file = "models/preprocess_info_logistic.pkl"
    use_scaler = True
else:
    model_file = "models/churn_model_naive.pkl"
    info_file = "models/preprocess_info_naive.pkl"
    use_scaler = False

# Load model and info
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(info_file, 'rb') as f:
        info = pickle.load(f)
    if use_scaler:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

categorical_cols = info['categorical_cols']
feature_names = info['feature_names']

# Title
st.title("📡 Telco Customer Churn Predictor")
st.markdown(f"**Active Model:** {model_choice}")
st.markdown("---")

# Form
with st.form("customer_form", clear_on_submit=False):
    st.markdown("### Customer Details")
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

    submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

if submitted:
    total_charges = tenure * monthly
    st.info(f"**Auto-Calculated Total Charges**: ${total_charges:.2f}")

    # Prepare input
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
        input_final = input_encoded.values if hasattr(input_encoded, 'values') else input_encoded
    
    # Prediction
    probability = model.predict_proba(input_final)[0][1]
    prediction = model.predict(input_final)[0]
    
    st.markdown("---")
    
    if prediction == 1:
        st.error("🚨 **HIGH RISK OF CHURN**")
        st.warning(f"Churn Probability: {probability:.2%}")
        st.info("💡 **Immediate Action Required**: Offer discount, long-term contract, or personal call!")
    else:
        st.success("✅ **LOW RISK – Customer Likely to Stay**")
        st.success(f"Churn Probability: {probability:.2%}")
        st.info("👍 Excellent! Continue providing great service.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>Made with ❤️ by Azam Khan | Powered by Streamlit</p>", unsafe_allow_html=True)
