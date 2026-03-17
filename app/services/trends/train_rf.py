import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_random_forest():
    print("Loading preprocessed data for Random Forest (Bagging)...")
    X_train = pd.read_csv('data/ml_ready/train_features.csv')
    y_train = pd.read_csv('data/ml_ready/train_target.csv').values.ravel()
    X_test = pd.read_csv('data/ml_ready/test_features.csv')
    y_test = pd.read_csv('data/ml_ready/test_target.csv').values.ravel()
    
    # Simple Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    
    print("Training Random Forest...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Random Forest Performance ---")
    print(f"MAE (Log): {mae:.4f}")
    print(f"MSE (Log): {mse:.4f}")
    print(f"R2 (Log): {r2:.4f}")
    
    os.makedirs('models/rf', exist_ok=True)
    joblib.dump(model, 'models/rf/cutoff_rf_v1.joblib')
    
    with open('research/rf_metrics.txt', 'w') as f:
        f.write(f"R2: {r2}\nMAE: {mae}\nMSE: {mse}\n")

if __name__ == "__main__":
    train_random_forest()
