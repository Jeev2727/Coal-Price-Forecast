import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.regression.linear_model import OLSResults
from sqlalchemy import create_engine
from urllib.parse import quote
import matplotlib.pyplot as plt

# Load trained model
model = OLSResults.load("/Users/mandalajeevan/coal_price_prediction_model.pkl")

# Streamlit UI
st.title("Coal Price Forecasting")

st.sidebar.title("Input External Factors")

# User Inputs for latest values of external factors
crude_oil = st.sidebar.number_input("Crude Oil Price (USD)", min_value=0.0, step=0.1, value=80.0)
brent_oil = st.sidebar.number_input("Brent Oil Price (USD)", min_value=0.0, step=0.1, value=85.0)
dubai_crude = st.sidebar.number_input("Dubai Crude Price (USD)", min_value=0.0, step=0.1, value=83.0)
dutch_ttf = st.sidebar.number_input("Dutch TTF Price (USD)", min_value=0.0, step=0.1, value=30.0)
natural_gas = st.sidebar.number_input("Natural Gas Price (USD)", min_value=0.0, step=0.1, value=3.5)

# Database Credentials Input
st.sidebar.subheader("Database Credentials")
user = st.sidebar.text_input("Username", "Type Here")
pw = st.sidebar.text_input("Password", "Type Here", type='password')
db = st.sidebar.text_input("Database", "Type Here")

# Predict Button
if st.button("Predict Next 30 Days Coal Prices"):

    future_days = 30
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=future_days)

    # Create input DataFrame for prediction (same external factors for next 30 days)
    future_external_df = pd.DataFrame({
        "Crude Oil_Price": [crude_oil] * future_days,
        "Brent Oil_Price": [brent_oil] * future_days,
        "Dubai Crude_Price": [dubai_crude] * future_days,
        "Dutch TTF_Price": [dutch_ttf] * future_days,
        "Natural Gas_Price": [natural_gas] * future_days
    })

    # Predict coal prices
    future_predictions = model.predict(future_external_df)

    # Create DataFrame with results
    future_df = pd.DataFrame(future_predictions, columns=[
        "Coal Richards Bay 4800kcal", "Coal Richards Bay 5500kcal", "Coal Richards Bay 5700kcal",
        "Coal Richards Bay 6000kcal", "Coal India 5500kcal"
    ])
    future_df["Date"] = future_dates

    # Display predictions
    st.write("### Predicted Coal Prices (Next 30 Days)")
    st.dataframe(future_df)

    # Save to database
    if user and pw and db:
        try:
            engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")
            future_df.to_sql('forecast_results', con=engine, if_exists='replace', index=False, chunksize=1000)
            st.success("Forecast saved to database successfully!")
        except Exception as e:
            st.error(f"Database Error: {e}")

    # Plot results
    st.write("### Prediction Trends")
    plt.figure(figsize=(12, 6))
    for col in future_df.columns[:-1]:
        plt.plot(future_df["Date"], future_df[col], label=col)
    
    plt.xlabel("Date")
    plt.ylabel("Price (USD/t)")
    plt.title("Coal Price Forecast for Next 30 Days")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    
    st.pyplot(plt)

