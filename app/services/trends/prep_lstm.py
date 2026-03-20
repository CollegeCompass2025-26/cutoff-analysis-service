import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv("c:/cutoff-analysis-service/.env")

def prepare_lstm_sequences():
    print("Fetching sequential round data from ml_features_v3...")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    
    # We pull the data ordered exactly by year and round to form a natural time series
    query = """
    SELECT 
        college_name, course_name, specialization, exam_name, category, quota, gender,
        year, round, closing_rank 
    FROM ml_features_v3 
    WHERE closing_rank > 0
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 1. Normalize Round Names
    # Round columns contain things like 'Round 1', '1', 'I', 'II', '6'. We must map them to integers.
    def parse_round(r):
        r = str(r).upper().replace('ROUND', '').strip()
        if r in ['1', 'I']: return 1
        if r in ['2', 'II']: return 2
        if r in ['3', 'III']: return 3
        if r in ['4', 'IV']: return 4
        if r in ['5', 'V']: return 5
        if r in ['6', 'VI']: return 6
        return 1 # Default
        
    df['round_num'] = df['round'].apply(parse_round)
    
    # Sort temporally
    df = df.fillna('N/A')
    group_cols = ['college_name', 'course_name', 'specialization', 'exam_name', 'category', 'quota', 'gender']
    df = df.sort_values(by=group_cols + ['year', 'round_num'])
    print(f"Processed {len(df)} records into {df['round_num'].nunique()} distinct round phases.")
    
    # 2. Build Sequences
    # We want a sequence of the last N historical rounds across any years to predict the next round.
    # Group by the exact seat pool
    sequences = []
    targets = []
    
    # We define a fixed lookback window. Say, look at the last 4 cutoff events to predict the 5th.
    LOOKBACK = 4
    
    grouped = df.groupby(group_cols)
    
    for _, group in grouped:
        ranks = group['closing_rank'].values
        # Only use groups that have enough history
        if len(ranks) > LOOKBACK:
            for i in range(len(ranks) - LOOKBACK):
                seq = ranks[i : i + LOOKBACK]
                target = ranks[i + LOOKBACK]
                
                sequences.append(seq)
                targets.append(target)
                
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Total temporal cross-round sequences generated: {len(X)}")
    
    # Reshape for LSTM: [samples, time_steps, features]
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Log transform for stability
    X_seq = np.log1p(X_seq)
    y_log = np.log1p(y)
    
    os.makedirs('data/ml_ready', exist_ok=True)
    np.save('data/ml_ready/X_lstm.npy', X_seq)
    np.save('data/ml_ready/y_lstm.npy', y_log)
    print("LSTM tensor generation completed successfully.")
    
    return X_seq, y_log

if __name__ == "__main__":
    prepare_lstm_sequences()
