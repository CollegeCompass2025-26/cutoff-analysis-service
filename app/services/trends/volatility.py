import pandas as pd
import numpy as np
import os
import joblib

def calculate_volatility():
    print("Calculating institutional volatility indices...")
    # We use the standard deviation of logs of ranks over years for each pair
    X = np.load('data/ml_ready/X_lstm.npy') # [samples, 3, 1]
    
    # Volatility = Std Dev of the 3-year sequence
    # This measures how much the rank fluctuates year-over-year
    vols = np.std(X.squeeze(), axis=1)
    
    # Normalize to 0-100 score
    max_vol = np.percentile(vols, 95) # Cap at 95th percentile to avoid outliers
    vol_scores = (vols / max_vol) * 100
    vol_scores = np.clip(vol_scores, 0, 100)
    
    np.save('data/ml_ready/volatilities.npy', vol_scores)
    print(f"Volatility indices calculated for {len(vol_scores)} institution-branch pairs.")
    
    # Generate high/medium/low tags
    # High > 60, Medium 30-60, Low < 30
    
if __name__ == "__main__":
    calculate_volatility()
