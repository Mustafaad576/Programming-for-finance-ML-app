import pandas as pd

def load_data(file):
    return pd.read_csv(file)

def preprocess_data(df):
    df = df.dropna()
    return df

def feature_engineering(df):
    return df  # Placeholder

def split_data(df):
    from sklearn.model_selection import train_test_split
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42)
