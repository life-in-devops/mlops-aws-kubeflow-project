import joblib
import pandas as pd

from preprocess import load_data, clean_data

# Load model + feature schema
model = joblib.load("model.joblib")
features = joblib.load("features.joblib")

# Load raw data
df = load_data("data/churn.csv")
df = clean_data(df)

# Take one sample
sample = df.drop("Churn", axis=1).iloc[[0]]

# Apply encoding
sample = pd.get_dummies(sample)

# 🔷 Align with training features (CRITICAL FIX)
sample = sample.reindex(columns=features, fill_value=0)

# Predict
pred = model.predict(sample)

print("Prediction:", pred)