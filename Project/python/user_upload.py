# user_upload.py

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

def upload_and_predict(period):
    uploaded_file = st.file_uploader("Upload your CSV file here:", type=["csv"])
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        
        # Display uploaded data
        st.subheader("Uploaded data")
        st.write(user_data.tail())

        # Print column names
        st.write("Column Names:", user_data.columns.tolist())
        
        # Check if 'Date' and 'Close' columns exist
        if 'Date' in user_data.columns and 'Close' in user_data.columns:
            # Prepare data for prediction
            user_df = user_data[['Date', 'Close']]
            user_df = user_df.rename(columns={"Date": "ds", "Close": "y"})
            
            # Make prediction
            m_user = Prophet()
            m_user.fit(user_df)
            future_user = m_user.make_future_dataframe(periods=period)
            forecast_user = m_user.predict(future_user)
            
            # Display forecasted data
            st.subheader("Forecasted data for uploaded file")
            st.write(forecast_user.tail())
            
            # Plot forecasted data
            fig_user = plot_plotly(m_user, forecast_user)
            st.plotly_chart(fig_user)
        else:
            st.error("Required columns 'Date' and 'Close' not found in the uploaded data. Please make sure your CSV file contains these columns.")
