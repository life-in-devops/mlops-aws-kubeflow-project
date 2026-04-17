import pandas as pd

df = pd.read_csv("data/churn.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())