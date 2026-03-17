import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_xgboost():
    print("Loading preprocessed data...")
    X_train = pd.read_csv('data/ml_ready/train_features.csv')
    y_train = pd.read_csv('data/ml_ready/train_target.csv').values.ravel()
    X_test = pd.read_csv('data/ml_ready/test_features.csv')
    y_test = pd.read_csv('data/ml_ready/test_target.csv').values.ravel()
    
    print("Initializing XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method='hist' # For faster training on CPU
    )
    
    print("Training model (this may take a few minutes)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # 1. Predictions
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)
    
    # 2. Metrics
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test, y_pred_log) # Metrics on log scale are often more stable for ranks
    
    print(f"\n--- Model Performance ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score (log scale): {r2:.4f}")
    
    # 3. Save Model
    os.makedirs('models/xgboost', exist_ok=True)
    joblib.dump(model, 'models/xgboost/cutoff_xgb_v1.joblib')
    
    # 4. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('research/xgboost_feature_importance.png')
    
    # 5. Export metrics for research paper
    with open('research/xgboost_metrics.txt', 'w') as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2: {r2}\n")

if __name__ == "__main__":
    train_xgboost()
