import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn


import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RealEstate_Project")

# ==============================
# 1. Load Cleaned Data
# ==============================

df = pd.read_csv("data/cleaned_data.csv")
df = pd.get_dummies(df, drop_first=True)
# Reduce dataset size to avoid memory error
df = df.sample(30000, random_state=42)


print("Data Loaded & Encoded Successfully!")

print("Data Loaded Successfully!")
print("Shape of Data:", df.shape)

# ==============================
# 2. Define Features and Targets
# ==============================

# Classification Target
X_class = df.drop(["Good_Investment", "Future_Price_5Y"], axis=1)
y_class = df["Good_Investment"]

# COMMON FEATURES
X = df.drop(["Good_Investment", "Future_Price_5Y"], axis=1)

# Regression Target
X_reg = df.drop(["Future_Price_5Y", "Good_Investment"], axis=1)
y_reg = df["Future_Price_5Y"]

# ==============================
# 3. Train-Test Split
# ==============================

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ==============================
# 4. Feature Scaling
# ==============================

scaler = StandardScaler()

Xc_train = scaler.fit_transform(Xc_train)
Xc_test = scaler.transform(Xc_test)

Xr_train = scaler.fit_transform(Xr_train)
Xr_test = scaler.transform(Xr_test)

# 5. Train Classification Model
# ==============================

with mlflow.start_run(run_name="RandomForest_Classifier"):

    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xc_train, yc_train)

    y_pred_class = clf.predict(Xc_test)
    acc = accuracy_score(yc_test, y_pred_class)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(clf, "classification_model")

    print("Classification Accuracy:", acc)


# ==============================
# 6. Train Regression Model
# ==============================

with mlflow.start_run(run_name="RandomForest_Regressor"):

    reg = RandomForestRegressor(random_state=42)
    reg.fit(Xr_train, yr_train)

    y_pred_reg = reg.predict(Xr_test)

    r2 = r2_score(yr_test, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(yr_test, y_pred_reg))

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(reg, "regression_model")

    print("Regression R2:", r2)
    print("Regression RMSE:", rmse)

# ==============================
# 7. Save Models
# ==============================

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(clf, "models/classification_model.pkl")
joblib.dump(reg, "models/regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns, "models/feature_columns.pkl")

print("\nModels & feature columns saved successfully!")