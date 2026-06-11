import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Real Estate Advisor", layout="centered")

# =========================================
# LOAD MODELS
# =========================================

@st.cache_resource
def load_models():
    clf             = joblib.load("models/classification_model.pkl")
    reg             = joblib.load("models/regression_model.pkl")
    scaler_class    = joblib.load("models/scaler_class.pkl")
    scaler_reg      = joblib.load("models/scaler_reg.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return clf, reg, scaler_class, scaler_reg, feature_columns

clf, reg, scaler_class, scaler_reg, feature_columns = load_models()

# City growth rates (same as training)
CITY_GROWTH = {
    "Mumbai": 0.085, "Bangalore": 0.095, "Pune": 0.090,
    "Hyderabad": 0.092, "New Delhi": 0.080, "Noida": 0.078,
    "Gurgaon": 0.082, "Chennai": 0.075, "Kolkata": 0.065,
    "Ahmedabad": 0.072, "Nashik": 0.072, "Nanded": 0.048,
    "Latur": 0.045, "Sambhaji Nagar": 0.055,
}
DEFAULT_GROWTH = 0.060

# =========================================
# UI
# =========================================

st.title("🏠 Real Estate Investment Predictor")
st.write("Enter your property details to get investment results:")

locations = ["Mumbai", "Pune", "Nashik", "Nanded", "Latur", "Sambhaji Nagar"]
selected_location = st.selectbox("Select Location:", options=locations)

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (Sqft)", min_value=100.0, step=50.0, value=1000.0)
    bedrooms = st.number_input("BHK", min_value=1, max_value=5, step=1, value=2)
with col2:
    current_price = st.number_input("Current Price (₹ in Lakhs)", min_value=1.0, step=1.0, value=50.0)
    age_of_property = st.number_input("Age of Property (Years)", min_value=0, max_value=50, step=1, value=5)



# =========================================
# PREDICTION
# =========================================

if st.button("📊 Predict Results"):

    # Build input DataFrame matching training columns
    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0

    if "Size_in_SqFt"    in input_data.columns: input_data.at[0, "Size_in_SqFt"]    = area
    if "BHK"             in input_data.columns: input_data.at[0, "BHK"]             = bedrooms
    if "Price_in_Lakhs"  in input_data.columns: input_data.at[0, "Price_in_Lakhs"]  = current_price
    if "Age_of_Property" in input_data.columns: input_data.at[0, "Age_of_Property"] = age_of_property
    

    city_col = f"City_{selected_location}"
    if city_col in input_data.columns:
        input_data.at[0, city_col] = 1

    input_data = input_data[feature_columns].astype(float)

    # Predictions
    input_scaled_class = scaler_class.transform(input_data)
    input_scaled_reg   = scaler_reg.transform(input_data)

    invest_pred  = clf.predict(input_scaled_class)[0]
    future_price = reg.predict(input_scaled_reg)[0]

    # =========================================
    # MANUAL GROWTH CALCULATION (for chart)
    # =========================================

    base_rate = CITY_GROWTH.get(selected_location, DEFAULT_GROWTH)

    # Adjustments (same logic as training)
    rate = base_rate
    if age_of_property <= 2:    rate += 0.010
    elif age_of_property <= 5:  rate += 0.005
    elif age_of_property >= 20: rate -= 0.010
   
    rate = max(0.02, rate)

    annual_growth_pct = round(rate * 100, 2)

    # =========================================
    # RESULTS
    # =========================================

    st.divider()
    st.subheader(f"Results for {selected_location}")

    col_a, col_b = st.columns(2)
    with col_a:
        if invest_pred == 1:
            st.success("✅ Good Investment")
        else:
            st.error("❌ High Risk Investment")
    with col_b:
        st.metric("Predicted Future Price (5Y)", f"₹ {future_price:,.2f} Lakhs")

    # Growth rate info
    st.info(f"📈 Estimated Annual Growth Rate for **{selected_location}**: **{annual_growth_pct}% per year**")

    # =========================================
    # YEAR-BY-YEAR TREND CHART
    # =========================================

    st.write("### 📈 Price Appreciation Trend")

    years       = list(range(6))
    price_trend = [round(current_price * ((1 + rate) ** y), 2) for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=price_trend,
        mode='lines+markers',
        name='Price Trend',
        line=dict(color='green', width=3),
        marker=dict(size=10),
        text=[f"₹{p:,.1f}L" for p in price_trend],
        textposition="top center"
    ))
    fig.update_layout(
        xaxis_title="Years",
        yaxis_title="Price (₹ Lakhs)",
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================================
    # PRICE GAIN SUMMARY
    # =========================================

    gain = future_price - current_price
    gain_pct = (gain / current_price) * 100

    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Current Price",      f"₹{current_price:,.1f}L")
    col_y.metric("Price After 5 Years", f"₹{future_price:,.1f}L", f"+₹{gain:,.1f}L")
    col_z.metric("Total Gain",          f"{gain_pct:.1f}%")

