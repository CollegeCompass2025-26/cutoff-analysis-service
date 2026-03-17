import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
import joblib

load_dotenv()

def prepare_lstm_sequences():
    print("Fetching raw data for sequence grouping...")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    # We need (College, Course, Category) groups that have multiple years of data
    query = """
    SELECT college_name, course_name, category, year, closing_rank 
    FROM raw_cutoffs 
    WHERE closing_rank > 0
    ORDER BY college_name, course_name, category, year
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Processing {len(df)} records into sequences...")
    
    # 1. Pivot to get years as columns
    pivot_df = df.pivot_table(
        index=['college_name', 'course_name', 'category'],
        columns='year',
        values='closing_rank'
    ).reset_index()
    
    # We want sequences of 2021, 2022, 2023 to predict 2024
    # Fill missing years with median or nearest neighbor if needed, but for now 
    # let's only take rows that have most of the data.
    
    # Drop rows where 2024 is missing (our target)
    seq_df = pivot_df.dropna(subset=[2024])
    
    # Filling missing historical years with the 2024 value as a baseline
    seq_df[2021] = seq_df[2021].fillna(seq_df[2024])
    seq_df[2022] = seq_df[2022].fillna(seq_df[2024])
    seq_df[2023] = seq_df[2023].fillna(seq_df[2024])
    
    X = seq_df[[2021, 2022, 2023]].values
    y = seq_df[2024].values
    
    # Reshape for LSTM: [samples, time_steps, features]
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Log transform
    X_seq = np.log1p(X_seq)
    y_log = np.log1p(y)
    
    print(f"Total sequences created: {len(X_seq)}")
    
    os.makedirs('data/ml_ready', exist_ok=True)
    np.save('data/ml_ready/X_lstm.npy', X_seq)
    np.save('data/ml_ready/y_lstm.npy', y_log)
    
    return X_seq, y_log

if __name__ == "__main__":
    prepare_lstm_sequences()
