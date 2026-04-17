from fastapi import FastAPI
import joblib
import pandas as pd

# from preprocess import clean_data
from src.preprocess import clean_data

app = FastAPI()

# 🔷 Load artifacts once at startup
model = joblib.load("model.joblib")
features = joblib.load("features.joblib")


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: dict):

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Apply same preprocessing
    df = clean_data(df)

    # Encode
    df = pd.get_dummies(df)

    # 🔷 Align with training features (CRITICAL)
    df = df.reindex(columns=features, fill_value=0)

    # Predict
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }