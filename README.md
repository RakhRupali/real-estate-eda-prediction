
# Real Estate Investment Advisor 🏠

**Description:** This project focuses on the Exploratory Data Analysis (EDA) and price prediction of real estate properties in India. It aims to provide data-driven insights for property investment decisions by predicting prices based on key features like location, size, and amenities.

---

## 🚀 Project Overview
This repository contains a complete end-to-end machine learning pipeline:
1.  **Data Exploration:** Analyzing trends and patterns in housing data.
2.  **Model Training:** Building and saving predictive models (Regression and Classification).
3.  **Interactive Dashboard:** A user-friendly Streamlit interface for real-time predictions.

---

## 🛠️ Project Structure
* **`app.py`**: The main Streamlit application script.
* **`model_training.py`**: Script used for training and saving the ML models.
* **`Notebooks/eda_exploration.ipynb`**: Detailed analysis and visualization of the dataset.
* **`models/`**: Contains trained files like `regression_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`.
* **`data/`**: Includes `india_housing_prices.csv` and `cleaned_data.csv`.
* **`mlflow.db`**: Database for tracking machine learning experiments.

---

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Plotly
* **Framework:** Streamlit
* **Experiment Tracking:** MLflow

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
  
   git clone <your-repository-link>
   cd Project4
Install required dependencies:


pip install -r requirements.txt
Run the Streamlit app:


streamlit run app.py
✨ Key Features

Interactive Visuals: View price trends over time using Plotly charts.

Instant Prediction: Enter property details to get an estimated market price instantly.

Data Cleaning Pipeline: Automated preprocessing of raw housing data.

Author: Rupali Shrinivas Kendre
