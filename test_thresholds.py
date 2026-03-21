import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import psycopg2
from dotenv import load_dotenv

# Set path
sys.path.append(os.getcwd())
load_dotenv("c:/cutoff-analysis-service/.env")

from app.services.trends.ensemble import CutoffEnsemble

def test_strategies():
    print("Testing Threshold Strategies...")
    ensemble = CutoffEnsemble()
    
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        # Sample more records for statistical significance
        query = "SELECT * FROM ml_features_v3 WHERE closing_rank > 0 ORDER BY RANDOM() LIMIT 500"
        df_test = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        print(f"DATABASE FETCH FAILED: {e}")
        return

    results = []
    
    # Define strategies
    strategies = [
        {"name": "Current (Fixed 500)", "func": lambda p, u: ensemble.get_risk_assessment(p, u)},
        {"name": "Percentage (5%)", "func": lambda p, u: "SAFE" if (u - p) < -0.05*p else ("MODERATE" if (u - p) < 0.05*p else "RISKY")},
        {"name": "Percentage (10%)", "func": lambda p, u: "SAFE" if (u - p) < -0.10*p else ("MODERATE" if (u - p) < 0.10*p else "RISKY")},
        {"name": "Log-Scale (0.1)", "func": lambda p, u: "SAFE" if (np.log1p(u) - np.log1p(p)) < -0.1 else ("MODERATE" if (np.log1p(u) - np.log1p(p)) < 0.1 else "RISKY")}
    ]

    for strategy in strategies:
        y_true = []
        y_pred = []
        
        for _, row in df_test.iterrows():
            actual = float(row['closing_rank'])
            # Mock historical seq
            history = [actual * 0.95] * 4
            
            try:
                pred = ensemble.get_prediction({'college_name': row['college_name'], 'course_name': row['course_name'], 'exam_name': row['exam_name'], 'category': row['category'], 'year': row['year']}, history_seq=history)
                # Simulated User Rank: We need a ground truth verdict. 
                # Let's assume the user IS the actual rank (Self-test)
                # Or better: let's use a distribution of user ranks around the actual.
                user_rank = actual * 1.02 # Slightly above actual
                
                y_pred.append(strategy["func"](pred, user_rank))
                y_true.append(strategy["func"](actual, user_rank))
            except:
                continue
                
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        print(f"{strategy['name']}: Accuracy={acc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    test_strategies()
