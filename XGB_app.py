import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Custom Styling (Optional)
# ---------------------------------
st.markdown("""
    <style>
    /* General button style */
    .stButton>button {
        background-color: #0078D7;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        height: 3em;
        width: 100%;
    }
    /* Title color */
    .stApp header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ---------------------------------
# Load Artifacts
# ---------------------------------
# Load the 5 artifacts from your training script
try:
    num_imputer = joblib.load("num_imputer.pkl")
    cat_imputer = joblib.load("cat_imputer.pkl")
    ohe = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("xgb_model.pkl")
except FileNotFoundError:
    st.error("Model artifacts not found. Please ensure all 5 .pkl files are in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ---------------------------------
# Hardcoded Feature Lists
# ---------------------------------
# These lists must be in the *exact same order* as they were during training.

# Original numerical features the user will input
# (From the dataset, before any engineering)
original_numeric_features = [
    'account.length', 'voice.messages', 'intl.mins', 'intl.calls',
    'intl.charge', 'day.mins', 'day.calls', 'day.charge', 'eve.mins',
    'eve.calls', 'eve.charge', 'night.mins', 'night.calls',
    'night.charge', 'customer.calls'
]

# Original categorical features the user will input
original_categorical_features = ['state', 'voice.plan', 'intl.plan']

# This is the list of ALL numeric features (original + engineered)
# that the scaler was trained on.
numeric_cols = [
    'account.length', 'voice.messages', 'intl.mins', 'intl.calls',
    'intl.charge', 'day.mins', 'day.calls', 'day.charge', 'eve.mins',
    'eve.calls', 'eve.charge', 'night.mins', 'night.calls',
    'night.charge', 'customer.calls', 'total_charge', 'total_mins',
    'charge_per_min', 'total_calls', 'mins_per_call', 'pay_as_you_go_intl'
]

# This is the list of ALL categorical features (original + engineered)
# that the one-hot encoder was trained on.
categorical_cols = ['state', 'voice.plan', 'intl.plan', 'service_call_bin']

# List of states for the dropdown (from the original dataset)
states_list = [
    'KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI',
    'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC',
    'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR',
    'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC',
    'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND'
]

# ---------------------------------
# Helper Function for Binning
# ---------------------------------
def get_service_call_bin(calls):
    """Bins the customer service calls into categories."""
    if calls == 0:
        return '0_calls'
    elif calls <= 3:
        return '1-3_calls'
    else:
        return '4+_calls'

# ---------------------------------
# Prediction Function
# ---------------------------------
def make_prediction(user_data):
    """
    Takes user input, applies the full pipeline, and returns a prediction.
    """
    # 1. Convert user input to DataFrame
    df = pd.DataFrame([user_data])

    # 2. Apply Feature Engineering (must match training script)
    # 2a. Create 'total' features
    df['total_charge'] = df['day.charge'] + df['eve.charge'] + df['night.charge'] + df['intl.charge']
    df['total_mins'] = df['day.mins'] + df['eve.mins'] + df['night.mins'] + df['intl.mins']
    df['charge_per_min'] = df['total_charge'] / (df['total_mins'] + 1e-6)
    df['total_calls'] = df['day.calls'] + df['eve.calls'] + df['night.calls'] + df['intl.calls']
    df['mins_per_call'] = df['total_mins'] / (df['total_calls'] + 1e-6)
    
    # 2b. Bin 'customer.calls'
    df['service_call_bin'] = df['customer.calls'].apply(get_service_call_bin)
    
    # 2c. Create 'pay_as_you_go_intl'
    df['pay_as_you_go_intl'] = (
        (df['intl.plan'] == 'no') & (df['intl.mins'] > 0)
    ).astype(int)

    # 3. Separate into numeric and categorical (in the correct order)
    try:
        num_data = df[numeric_cols]
        cat_data = df[categorical_cols]
    except KeyError as e:
        st.error(f"Feature mismatch during pipeline: {e}")
        return None, None

    # 4. Apply Preprocessing Pipeline
    num_data_imputed = num_imputer.transform(num_data)
    cat_data_imputed = cat_imputer.transform(cat_data)
    cat_data_encoded = ohe.transform(cat_data_imputed)
    
    # 5. Combine and Scale
    X_processed = np.hstack([num_data_imputed, cat_data_encoded])
    X_scaled = scaler.transform(X_processed)
    
    # 6. Make Prediction
    try:
        prediction_proba = model.predict_proba(X_scaled)[0]
        prediction = model.predict(X_scaled)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

    # 'prediction' will be 0 or 1, 'prediction_proba' is [prob_0, prob_1]
    return prediction, prediction_proba[1] # Return 0/1 and prob of '1' (Churn)

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("📡 Telecom Customer Churn Predictor")
st.markdown("""
This app predicts whether a telecom customer is likely to **Churn** (leave the service) 
using your 98% accurate XGBoost model. 

Provide the customer's details in the sidebar to get a prediction.
""")

st.sidebar.header("Customer Details")

# --- Create Sidebar Inputs ---
user_input = {}

# Categorical Inputs
st.sidebar.subheader("Account Info")
user_input['state'] = st.sidebar.selectbox("State", states_list)
user_input['account.length'] = st.sidebar.slider("Account Length (days)", 0, 250, 100)

# Voice Plan
st.sidebar.subheader("Voice Plan")
user_input['voice.plan'] = st.sidebar.radio("Voice Mail Plan", ['yes', 'no'], index=1)
user_input['voice.messages'] = st.sidebar.slider("Number of Voice Mails", 0, 60, 0)

# International Plan
st.sidebar.subheader("International Plan")
user_input['intl.plan'] = st.sidebar.radio("International Plan", ['yes', 'no'], index=1)
user_input['intl.mins'] = st.sidebar.number_input("International Mins", 0.0, 25.0, 10.0, 0.1)
user_input['intl.calls'] = st.sidebar.number_input("International Calls", 0, 20, 3)
user_input['intl.charge'] = st.sidebar.number_input("International Charge", 0.0, 7.0, 2.7, 0.1)

# Day Usage
st.sidebar.subheader("Daytime Usage")
user_input['day.mins'] = st.sidebar.number_input("Day Mins", 0.0, 360.0, 180.0, 1.0)
user_input['day.calls'] = st.sidebar.number_input("Day Calls", 0, 170, 100)
user_input['day.charge'] = st.sidebar.number_input("Day Charge", 0.0, 60.0, 30.0, 0.1)

# Evening Usage
st.sidebar.subheader("Evening Usage")
user_input['eve.mins'] = st.sidebar.number_input("Eve Mins", 0.0, 370.0, 200.0, 1.0)
user_input['eve.calls'] = st.sidebar.number_input("Eve Calls", 0, 170, 100)
user_input['eve.charge'] = st.sidebar.number_input("Eve Charge", 0.0, 31.0, 17.0, 0.1)

# Night Usage
st.sidebar.subheader("Night Usage")
user_input['night.mins'] = st.sidebar.number_input("Night Mins", 0.0, 400.0, 200.0, 1.0)
user_input['night.calls'] = st.sidebar.number_input("Night Calls", 0, 180, 100)
user_input['night.charge'] = st.sidebar.number_input("Night Charge", 0.0, 18.0, 9.0, 0.1)

# Customer Service
st.sidebar.subheader("Customer Service")
user_input['customer.calls'] = st.sidebar.slider("Customer Service Calls", 0, 10, 1)

# --- Custom Threshold Slider (Optional) ---
threshold = st.sidebar.slider("Churn Probability Threshold", 0.1, 0.9, 0.5, 0.05)

# --- Prediction Button and Output ---
st.divider()

if st.button("Predict Churn", type="primary", use_container_width=True):
    # Make prediction
    prediction, probability = make_prediction(user_input)

    if probability is not None:
        churn_label = 1 if probability >= threshold else 0

        if prediction is not None:
            churn_probability_percent = probability * 100

            # Display output
            if churn_label == 1:  # 1 means 'yes' (Churn)
                st.error("Prediction: **Customer WILL Churn**")
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%")
                st.warning("Action Recommended: Proactively engage this customer with retention offers.")
            else:  # 0 means 'no' (Not Churn)
                st.success("Prediction: **Customer will NOT Churn**")
                st.metric(label="Churn Probability", value=f"{churn_probability_percent:.2f}%")
                st.info("Action Recommended: Monitor customer, but no immediate retention action needed.")

            # Save predictions for analysis
            with open("predictions_log.csv", "a") as f:
                f.write(f"{user_input}, Prediction: {churn_label}, Probability: {probability:.2f}\n")

            # Show a breakdown of the input data
            with st.expander("Show Customer Input Summary"):
                st.json(user_input)
