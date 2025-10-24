import joblib
import numpy as np
import pandas as pd

# Load saved artifacts
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("xgb_model.pkl")

# Set your best threshold (update if needed)
THRESHOLD = 0.40   # <-- YOU CAN CHANGE THIS BASED ON ROC CURVE

def preprocess_input(df):
    # Apply encoder
    encoded = encoder.transform(df[encoder.feature_names_in_])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # Scale numerical columns
    numeric_cols = scaler.feature_names_in_
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    final_df = pd.concat([df[numeric_cols], encoded_df], axis=1)
    return final_df

def predict_churn(raw_df):
    processed = preprocess_input(raw_df)
    prob = model.predict_proba(processed)[:, 1]

    # Apply threshold
    prediction = (prob >= THRESHOLD).astype(int)
    return prediction, prob
