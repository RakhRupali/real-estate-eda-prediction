import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==============================
# 1. Load Saved Models & Columns
# ==============================
@st.cache_resource
def load_models():
    # Aapke folder structure ke hisaab se paths set hain
    clf = joblib.load("models/classification_model.pkl")
    reg = joblib.load("models/regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return clf, reg, scaler, feature_columns

clf, reg, scaler, feature_columns = load_models()

# ==============================
# 2. Streamlit UI Setup
# ==============================
st.set_page_config(page_title="Real Estate Advisor", layout="centered")
st.title("🏠 Real Estate Investment Predictor")
st.write("Apne property ki details bhariye aur investment result dekhiye:")

# Nanded ke famous areas ka dropdown
locations = ["Mumbai", "Pune", "Nashik", "Nanded", "Latur", "Sambhaji Nagar"]
selected_location = st.selectbox("Location select karein:", options=locations)

# Input fields ko do columns mein divide kiya hai
ui_col1, ui_col2 = st.columns(2)

with ui_col1:
    area = st.number_input("Area (Sqft)", min_value=100.0, step=50.0, value=1000.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, step=1, value=2)

with ui_col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, step=1, value=2)
    current_price = st.number_input("Current Price (₹)", min_value=100000.0, step=50000.0, value=5000000.0)

# ==============================
# 3. Prediction & Logic
# ==============================

if st.button("📊 Predict Results"):
    # Khali dataframe banana jo model ke columns se match kare
    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0  # Sab values pehle 0 set karein

    # Basic features fill karein
    if "Area" in input_data.columns: input_data.at[0, "Area"] = area
    if "Bedrooms" in input_data.columns: input_data.at[0, "Bedrooms"] = bedrooms
    if "Bathrooms" in input_data.columns: input_data.at[0, "Bathrooms"] = bathrooms
    if "Price" in input_data.columns: input_data.at[0, "Price"] = current_price

    # Location mapping (Assumption: Training mein 'Location_Name' format tha)
    loc_col = f"Location_{selected_location}"
    if loc_col in input_data.columns:
        input_data.at[0, loc_col] = 1

    # Data scale karein aur predict karein
    input_data = input_data[feature_columns] # Sequence check
    input_scaled = scaler.transform(input_data)

    invest_pred = clf.predict(input_scaled)[0]
    future_price = reg.predict(input_scaled)[0]

    # ==============================
    # 4. Results Display
    # ==============================
    st.divider()
    st.subheader(f"Results for {selected_location}")

    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if invest_pred == 1:
            st.success("✅ Good Investment")
        else:
            st.error("❌ High Risk Investment")

    with res_col2:
        st.metric(label="Predicted Future Price (5Y)", value=f"₹ {future_price:,.2f}")

    # --- Graph Section ---
    st.write("### 📈 Appreciation Trend")
    years = [0, 1, 2, 3, 4, 5]
    price_trend = np.linspace(current_price, future_price, 6)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, 
        y=price_trend, 
        mode='lines+markers',
        line=dict(color='#2ecc71' if invest_pred == 1 else '#e74c3c', width=4),
        marker=dict(size=10),
        name="Value over time"
    ))

    fig.update_layout(
        xaxis_title="Years",
        yaxis_title="Price (₹)",
        template="plotly_white",
        yaxis=dict(tickformat=",.0f", tickprefix="₹"), # Currency format fix
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)