import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/india_housing_prices.csv")
print("Raw data loaded:", df.shape)

# Create Amenities_Count from Amenities string
df["Amenities_Count"] = df["Amenities"].apply(
    lambda x: len(str(x).split(",")) if pd.notna(x) and str(x) != "" else 0
)

city_growth_rates = {
    "Mumbai": 0.085, "Bangalore": 0.095, "Pune": 0.090,
    "Hyderabad": 0.092, "New Delhi": 0.080, "Noida": 0.078,
    "Gurgaon": 0.082, "Chennai": 0.075, "Kolkata": 0.065,
    "Ahmedabad": 0.072, "Surat": 0.070, "Kochi": 0.078,
    "Jaipur": 0.068, "Lucknow": 0.062, "Indore": 0.070,
    "Nagpur": 0.060, "Coimbatore": 0.065, "Bhopal": 0.058,
    "Vishakhapatnam": 0.072, "Warangal": 0.055,
    "Vijayawada": 0.060, "Mysore": 0.065, "Mangalore": 0.062,
    "Faridabad": 0.070, "Guwahati": 0.058, "Bhubaneswar": 0.065,
    "Raipur": 0.055, "Patna": 0.050, "Ranchi": 0.052,
    "Amritsar": 0.055, "Ludhiana": 0.058, "Dehradun": 0.062,
    "Haridwar": 0.055, "Jodhpur": 0.052, "Jamshedpur": 0.050,
    "Durgapur": 0.048, "Silchar": 0.045, "Bilaspur": 0.045,
    "Gaya": 0.042, "Cuttack": 0.048, "Dwarka": 0.055,
    "Trivandrum": 0.065,
    # App cities
    "Nashik": 0.072, "Nanded": 0.048, "Latur": 0.045,
    "Sambhaji Nagar": 0.055,
}
DEFAULT_GROWTH = 0.060

np.random.seed(42)

def calc_future_price(row):
    rate = city_growth_rates.get(row["City"], DEFAULT_GROWTH)
    if row["Age_of_Property"] <= 2: rate += 0.010
    elif row["Age_of_Property"] <= 5: rate += 0.005
    elif row["Age_of_Property"] >= 20: rate -= 0.010
    if row["Nearby_Schools"] >= 8: rate += 0.005
    if row["Nearby_Hospitals"] >= 8: rate += 0.005
    if row["Amenities_Count"] >= 4: rate += 0.005
    elif row["Amenities_Count"] <= 1: rate -= 0.005
    noise = np.random.normal(0, 0.005)
    rate = max(0.02, rate + noise)
    return round(row["Price_in_Lakhs"] * ((1 + rate) ** 5), 2)

def calc_good_investment(row):
    score = 0
    rate = city_growth_rates.get(row["City"], DEFAULT_GROWTH)
    if rate >= 0.080: score += 3
    elif rate >= 0.065: score += 2
    elif rate >= 0.050: score += 1
    ppsf = row["Price_in_Lakhs"] / max(row["Size_in_SqFt"], 1)
    if ppsf < 0.05: score += 2
    elif ppsf < 0.10: score += 1
    elif ppsf > 0.20: score -= 1
    if row["Age_of_Property"] <= 5: score += 2
    elif row["Age_of_Property"] >= 20: score -= 2
    if row["Nearby_Schools"] >= 7: score += 1
    if row["Nearby_Hospitals"] >= 7: score += 1
    if row["Amenities_Count"] >= 4: score += 1
    if row["BHK"] in [2, 3]: score += 1
    return 1 if score >= 5 else 0

df["Future_Price_5Y"] = df.apply(calc_future_price, axis=1)
df["Good_Investment"] = df.apply(calc_good_investment, axis=1)

final_df = df[[
    "Size_in_SqFt", "BHK", "Price_in_Lakhs", "City",
    "Nearby_Schools", "Nearby_Hospitals", "Amenities_Count",
    "Age_of_Property", "Good_Investment", "Future_Price_5Y"
]]

os.makedirs("data", exist_ok=True)
final_df.to_csv("data/final_training_data.csv", index=False)

print("Good Investment distribution:", df["Good_Investment"].value_counts().to_dict())
print("Future Price stats:\n", df["Future_Price_5Y"].describe())
print("\nCorrelation with Future_Price_5Y:")
print(final_df[["Size_in_SqFt","BHK","Price_in_Lakhs","Nearby_Schools","Nearby_Hospitals","Amenities_Count","Age_of_Property","Future_Price_5Y"]].corr()["Future_Price_5Y"])
print("\nSaved: data/final_training_data.csv")
