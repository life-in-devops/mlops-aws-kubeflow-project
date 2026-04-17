import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()

    # Drop customerID (not useful)
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop nulls
    df.dropna(inplace=True)

    return df

def encode_data(df):
    df = pd.get_dummies(df)
    return df