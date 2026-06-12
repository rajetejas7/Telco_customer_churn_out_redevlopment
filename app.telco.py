import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration and Model Loading (done once when the app starts) ---
# Ensure these files are in the same directory as your streamlit_app.py
# or provide the correct path if they are in a subfolder.
# model = joblib.load('xgboost_model.joblib')
# scaler = joblib.load('scaler.joblib')

# Placeholder for model and scaler if files are not present during development
try:
    model = joblib.load('xgboost_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.warning("Model or scaler files not found. Using dummy objects for testing purposes.")
    class DummyModel:
        def predict(self, X): return np.zeros(X.shape[0])
        def predict_proba(self, X): return np.array([[1.0, 0.0]] * X.shape[0])
    class DummyScaler:
        def transform(self, X): return X
    model = DummyModel()
    scaler = DummyScaler()

TRAINING_COLUMNS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# --- Streamlit UI and Prediction Logic ---
st.title('Telco Customer Churn Prediction')

st.write('Enter customer details to predict churn:')

# --- Input widgets for all features ---

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.radio('Gender', ['Male', 'Female'])
    senior_citizen = st.checkbox('Senior Citizen')
    partner = st.checkbox('Partner')
    dependents = st.checkbox('Dependents')
    phone_service = st.checkbox('Phone Service')

with col2:
    tenure = st.slider('Tenure (months)', 0, 72, 36)
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])

with col3:
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    paperless_billing = st.checkbox('Paperless Billing')

monthly_charges = st.number_input('Monthly Charges', 0.0, 500.0, 50.0)
total_charges = st.number_input('Total Charges', 0.0, 10000.0, 1000.0)

contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Collect user input into a dictionary and map to TRAINING_COLUMNS
user_input = {
    'SeniorCitizen': 1 if senior_citizen else 0,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': 1 if gender == 'Male' else 0,
    'Partner_Yes': 1 if partner else 0,
    'Dependents_Yes': 1 if dependents else 0,
    'PhoneService_Yes': 1 if phone_service else 0,
    'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
    'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
    'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet_service == 'No' else 0,
    'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
    'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
    'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
    'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
    'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
    'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
    'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
    'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
    'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
    'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
    'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
    'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'PaperlessBilling_Yes': 1 if paperless_billing else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
}

# Create a DataFrame from user input
input_df = pd.DataFrame([user_input])

# Preprocessing steps (must match training preprocessing)
input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
input_df.fillna(0, inplace=True) # Or a more robust imputation

# Reindex to ensure all columns (and their order) match the training data
# This also handles any potential columns not explicitly set above, filling them with 0
input_df = input_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)

# Scale numerical features
scaled_features = scaler.transform(input_df)

if st.button('Predict Churn'):
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)[:, 1][0]

    if prediction[0] == 1:
        st.error(f'Prediction: Customer is likely to CHURN (Probability: {prediction_proba:.2f})')
    else:
        st.success(f'Prediction: Customer is NOT likely to churn (Probability: {prediction_proba:.2f})')
