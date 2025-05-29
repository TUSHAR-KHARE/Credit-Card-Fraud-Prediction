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

    st.download_button("Download Results as CSV", result.to_csv(index=False), "fraud_results.csv", "text/csv")

