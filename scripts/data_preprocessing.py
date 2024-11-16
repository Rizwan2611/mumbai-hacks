# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Load the dataset
    df = pd.read_csv('/Users/rizwansalmani/Desktop/hack/.venv/bin/python /Users/rizwansalmani/Desktop/hack/data/ML-Dataset.csv')
    # Convert the date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Extract relevant features for model training
    features = df[['Sales_Quantity', 'Inventory']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, scaler

if __name__ == "__main__":

    data, scaler = load_data('/Users/rizwansalmani/Desktop/hack/.venv/bin/python /Users/rizwansalmani/Desktop/hack/data/ML-Dataset.csv')
    print("Data loaded and scaled:", data[:5])

    
