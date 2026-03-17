import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

load_dotenv()

class DataPreprocessor:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.encoders = {}

    def fetch_data(self, exam_name=None):
        print(f"Fetching data from database...")
        conn = psycopg2.connect(self.db_url)
        query = "SELECT * FROM raw_cutoffs"
        if exam_name:
            query += f" WHERE exam_name = '{exam_name}'"
        
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Fetched {len(df)} records.")
        return df

    def prepare_features(self, df):
        print("Starting feature engineering...")
        
        # 1. Handle missing values
        df = df.copy()
        df['opening_rank'] = df['opening_rank'].fillna(df['closing_rank'])
        df['percentile'] = df['percentile'].fillna(0)
        
        # 2. Categorical Encoding
        categorical_cols = ['exam_name', 'round', 'college_name', 'course_name', 'category', 'quota', 'gender', 'institute_type']
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fill NAs with 'Unknown' before encoding
            df[col] = df[col].fillna('Unknown').astype(str)
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
            
        # 3. Target Variable
        # Ensure closing_rank is numeric and handle outliers
        df['closing_rank'] = pd.to_numeric(df['closing_rank'], errors='coerce')
        df = df[df['closing_rank'] > 0] # Rank must be positive
        
        # 4. Filter out any remaining NaNs in features
        df = df.dropna(subset=['closing_rank'] + categorical_cols)
        
        # We might want to use log transform for ranks as they can vary wildly
        df['log_closing_rank'] = np.log1p(df['closing_rank'])
        
        # Drop infs if any
        df = df[~np.isinf(df['log_closing_rank'])]
        
        # 5. Save Encoders
        os.makedirs('models/encoders', exist_ok=True)
        joblib.dump(self.encoders, 'models/encoders/label_encoders.joblib')
        
        features = categorical_cols + ['year']
        X = df[features]
        y = df['log_closing_rank']
        
        return X, y, features

if __name__ == "__main__":
    processor = DataPreprocessor()
    df = processor.fetch_data()
    X, y, feat_names = processor.prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset prepared. Train size: {len(X_train)}, Test size: {len(X_test)}")
    # Save processed data for training
    os.makedirs('data/ml_ready', exist_ok=True)
    X_train.to_csv('data/ml_ready/train_features.csv', index=False)
    y_train.to_csv('data/ml_ready/train_target.csv', index=False)
    X_test.to_csv('data/ml_ready/test_features.csv', index=False)
    y_test.to_csv('data/ml_ready/test_target.csv', index=False)
