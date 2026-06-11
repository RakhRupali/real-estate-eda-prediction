import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv("data/final_training_data.csv")
print("Data loaded:", df.shape)

# =========================================
# FEATURES & TARGETS
# =========================================
X = df[["Size_in_SqFt","BHK","Price_in_Lakhs","City",
        "Nearby_Schools","Nearby_Hospitals","Amenities_Count","Age_of_Property"]]

X = pd.get_dummies(X, columns=["City"])
feature_columns = X.columns.tolist()

y_class = df["Good_Investment"]
y_reg   = df["Future_Price_5Y"]

# =========================================
# TRAIN/TEST SPLIT
# =========================================
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg,   test_size=0.2, random_state=42)

# =========================================
# SCALE
# =========================================
scaler_class = StandardScaler()
scaler_reg   = StandardScaler()

X_train_c_scaled = scaler_class.fit_transform(X_train_c)
X_test_c_scaled  = scaler_class.transform(X_test_c)

X_train_r_scaled = scaler_reg.fit_transform(X_train_r)
X_test_r_scaled  = scaler_reg.transform(X_test_r)

# =========================================
# MODELS (more trees, better depth)
# =========================================
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

clf.fit(X_train_c_scaled, y_train_c)
reg.fit(X_train_r_scaled, y_train_r)
print("Models trained!")

# =========================================
# EVALUATE
# =========================================
acc = accuracy_score(y_test_c, clf.predict(X_test_c_scaled))
mae = mean_absolute_error(y_test_r, reg.predict(X_test_r_scaled))
print(f"Classification Accuracy: {acc:.4f}")
print(f"Regression MAE: {mae:.2f} Lakhs")

# =========================================
# FEATURE IMPORTANCE
# =========================================
print("\nTop Features (Regression):")
imp = pd.Series(reg.feature_importances_, index=feature_columns).sort_values(ascending=False)
print(imp.head(10))

# =========================================
# SAVE
# =========================================
os.makedirs("models", exist_ok=True)
joblib.dump(clf,             "models/classification_model.pkl")
joblib.dump(reg,             "models/regression_model.pkl")
joblib.dump(scaler_class,    "models/scaler_class.pkl")
joblib.dump(scaler_reg,      "models/scaler_reg.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")
print("\nAll models saved!")
