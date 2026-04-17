import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):

    # Drop customerID safely
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Handle TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Optional: handle target if present
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str)

    return df

def encode_data(df):
    df = pd.get_dummies(df)
    return df