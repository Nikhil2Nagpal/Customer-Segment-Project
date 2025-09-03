import streamlit as st 
import numpy as np
import pandas as pd
import joblib

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=100000, value=50)
total_spend = st.number_input("Total Spend (k$)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'NumWebPurchases': [num_web_purchases],           # ✅ Order match
        'Total_Spend': [total_spend],                     # ✅
        'NumStorePurchases': [num_store_purchases],       # ✅
        'NumWebVisitsMonth': [num_web_visits],  
        'Recency': [recency]
        })

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"The customer belongs to segment: {cluster}")

    #st.write("""
   # - Cluster 0: High budget, web visiotrs,
    #         - Cluster 1: high spend,
    #         - Cluster 2: web visitors""")