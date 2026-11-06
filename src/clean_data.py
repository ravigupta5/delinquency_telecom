import pandas as pd
import os

def load_data(file_path):
    """Load the raw dataset."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def clean_telecom_data(df):
    """Clean the dataframe based on EDA findings."""
    print("Cleaning data...")
    
    # Drop the phone number (ID) as it's not a predictive feature
    if 'msisdn' in df.columns:
        df = df.drop(columns=['msisdn'])
        
    # Drop pcircle if it is a constant value
    if 'pcircle' in df.columns and df['pcircle'].nunique() <= 1:
        df = df.drop(columns=['pcircle'])
        
    # For baseline modeling, we will drop the date column. 
    # (In a more advanced iteration, we could extract day/month features here).
    if 'pdate' in df.columns:
        df = df.drop(columns=['pdate'])

    # V2 Update: Drop rows where 'aon' (Age on Network) is negative or zero, as these are data anomalies
    if 'aon' in df.columns:
        print("V2 Update: Filtering anomalous 'aon' records...")
        df = df[df['aon'] > 0]
        
    return df

def save_data(df, output_path):
    """Save the cleaned data."""
    print(f"Saving cleaned data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/raw_data.csv"
    PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"
    
    # Execute pipeline
    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_telecom_data(raw_df)
    save_data(cleaned_df, PROCESSED_DATA_PATH)
    
    print(f"Cleaned dataset shape: {cleaned_df.shape}")