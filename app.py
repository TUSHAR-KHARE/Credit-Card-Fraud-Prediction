import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Credit Card Fraud Detection")
st.markdown("Upload transaction data to predict fraud risk.")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read uploaded data
    input_df = pd.read_csv(uploaded_file)

    # Drop the 'Class' column if it exists (target column not needed for prediction)
    if 'Class' in input_df.columns:
        input_df = input_df.drop(columns=['Class'])

    # Add 'Hour' column derived from 'Time'
    if 'Time' in input_df.columns:
        input_df['Hour'] = (input_df['Time'] // 3600) % 24

    # Ensure columns are in the correct order as used during training
    expected_features = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
        'V28', 'Amount', 'Hour'
    ]
    input_df = input_df[expected_features]

    # Scale features
    scaled_input = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[:, 1]

    # Show predictions
    result = pd.DataFrame({
        "Prediction": prediction,
        "Fraud Probability": probability
    })

    st.write("### Prediction Results")
    st.write(result)

    # Download button
    st.download_button(
        label="Download Results as CSV",
        data=result.to_csv(index=False),
        file_name="fraud_results.csv",
        mime="text/csv"
    )
