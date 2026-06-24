import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path="data/dataset.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        # Generate dummy data for testing if real dataset isn't present
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'age': rng.integers(18, 70, 1000),
            'income': rng.integers(20000, 150000, 1000),
            'loan_amount': rng.integers(1000, 50000, 1000),
            'credit_history': rng.choice([0, 1], 1000),
            'default': rng.choice([0, 1], 1000, p=[0.8, 0.2])
        })
        return df

def preprocess_data(df):
    df.fillna(df.median(), inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop('default', axis=1)
    y = df['default']
    return train_test_split(X, y, test_size=0.2, random_state=42)
