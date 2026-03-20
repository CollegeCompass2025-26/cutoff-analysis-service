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

    def fetch_data(self):
        print(f"Fetching data from ml_features_v2...")
        conn = psycopg2.connect(self.db_url)
        query = "SELECT * FROM ml_features_v2"
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"Fetched {len(df)} records.")
        return df

    def prepare_features(self, df):
        print("Starting feature engineering...")
        
        # 1. Handle missing values for target
        df = df.copy()
        df['cutoff_value'] = pd.to_numeric(df['cutoff_value'], errors='coerce')
        df = df.dropna(subset=['cutoff_value'])
        df = df[df['cutoff_value'] > 0]
        
        # Add new numeric features
        numeric_features = [
            'established_year', 'rating_hostel', 'rating_academic', 'rating_faculty', 
            'rating_infra', 'rating_placement', 'highest_package', 'avg_package', 
            'fees', 'duration_years'
        ]
        
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 2. Categorical Encoding
        categorical_cols = [
            'category', 'cutoff_type', 'college_name', 'state', 'city', 
            'typeofuni', 'course_name', 'specialization_name', 'exam_name'
        ]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].fillna('Unknown').astype(str)
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
            
        # 3. Target Variable (Log Scale)
        df['log_cutoff_value'] = np.log1p(df['cutoff_value'])
        df = df[~np.isinf(df['log_cutoff_value'])]
        
        # 4. Save Encoders
        os.makedirs('models/encoders', exist_ok=True)
        joblib.dump(self.encoders, 'models/encoders/label_encoders.joblib')
        
        features = categorical_cols + numeric_features + ['year']
        X = df[features]
        y = df['log_cutoff_value']
        
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
