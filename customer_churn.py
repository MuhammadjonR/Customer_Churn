import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
xgb_model = joblib.load(r"D:\\Projects\\ML_CustomerChurn\\xgb_model.pkl")

# Define encoding for categorical features (example encoding)
marital_status_map = {'Single': 0, 'Married': 1, 'Divorced': 2}
gender_map = {'Male': 0, 'Female': 1}

# Streamlit App
st.title("XGB Model Prediction App")

# File uploader for input data
uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    data = pd.read_csv(uploaded_file)

    # Display the data
    st.subheader("Uploaded Data")
    st.dataframe(data)

    # Encode categorical features
    data['MaritalStatus'] = data['MaritalStatus'].map(marital_status_map)
    data['Gender'] = data['Gender'].map(gender_map)

    # Ensure all required columns are present
    required_columns = ['Tenure', 'Complain', 'DaySinceLastOrder', 'CashbackAmount', 'MaritalStatus', 'Gender']
    if all(col in data.columns for col in required_columns):
        # Make predictions
        predictions = xgb_model.predict(data)

        # Add predictions to the DataFrame
        data['Prediction'] = predictions

        # Display predictions
        st.subheader("Predictions")
        st.dataframe(data)
    else:
        st.error(f"Missing required columns. Ensure the uploaded file has: {required_columns}")